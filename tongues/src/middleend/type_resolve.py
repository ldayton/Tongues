"""Shared type resolver for middleend passes."""

from __future__ import annotations

from ..taytsh.ast import (
    TBoolLit,
    TByteLit,
    TBytesLit,
    TCall,
    TExpr,
    TFieldAccess,
    TFloatLit,
    TForStmt,
    TIndex,
    TIntLit,
    TListLit,
    TMapLit,
    TNilLit,
    TRange,
    TRuneLit,
    TSetLit,
    TSlice,
    TStringLit,
    TTupleLit,
    TVar,
)
from ..taytsh.check import (
    BOOL_T,
    BYTE_T,
    BYTES_T,
    Checker,
    FLOAT_T,
    INT_T,
    ListT,
    MapT,
    NIL_T,
    RUNE_T,
    STRING_T,
    SetT,
    StructT,
    TupleT,
    Type,
    VOID_T,
    normalize_union,
    type_eq,
)


class TypeResolver:
    """Base type resolver — shared by scope and strings passes."""

    def __init__(self, locals: dict[str, Type], checker: Checker):
        self.locals = locals
        self.checker = checker

    def resolve(self, expr: TExpr) -> Type | None:
        if isinstance(expr, TVar):
            if expr.name in self.locals:
                return self.locals[expr.name]
            if expr.name in self.checker.functions:
                return self.checker.functions[expr.name]
            if expr.name in self.checker.types:
                return self.checker.types[expr.name]
            return None
        if isinstance(expr, TIntLit):
            return INT_T
        if isinstance(expr, TFloatLit):
            return FLOAT_T
        if isinstance(expr, TBoolLit):
            return BOOL_T
        if isinstance(expr, TByteLit):
            return BYTE_T
        if isinstance(expr, TStringLit):
            return STRING_T
        if isinstance(expr, TRuneLit):
            return RUNE_T
        if isinstance(expr, TBytesLit):
            return BYTES_T
        if isinstance(expr, TNilLit):
            return NIL_T
        if isinstance(expr, TListLit):
            if expr.elements:
                elem_t = self.resolve(expr.elements[0])
                if elem_t is not None:
                    return ListT(kind="list", element=elem_t)
            return None
        if isinstance(expr, TMapLit):
            if expr.entries:
                kt = self.resolve(expr.entries[0][0])
                vt = self.resolve(expr.entries[0][1])
                if kt is not None and vt is not None:
                    return MapT(kind="map", key=kt, value=vt)
            return None
        if isinstance(expr, TSetLit):
            if expr.elements:
                elem_t = self.resolve(expr.elements[0])
                if elem_t is not None:
                    return SetT(kind="set", element=elem_t)
            return None
        if isinstance(expr, TTupleLit):
            elems: list[Type] = []
            for e in expr.elements:
                t = self.resolve(e)
                if t is None:
                    return None
                elems.append(t)
            return TupleT(kind="tuple", elements=elems)
        if isinstance(expr, TCall):
            return self.resolve_call(expr)
        if isinstance(expr, TFieldAccess):
            obj_t = self.resolve(expr.obj)
            if isinstance(obj_t, StructT):
                if expr.field in obj_t.fields:
                    return obj_t.fields[expr.field]
            return None
        if isinstance(expr, TIndex):
            obj_t = self.resolve(expr.obj)
            if obj_t is not None:
                if isinstance(obj_t, ListT):
                    return obj_t.element
                if isinstance(obj_t, MapT):
                    return obj_t.value
                if type_eq(obj_t, STRING_T):
                    return RUNE_T
                if type_eq(obj_t, BYTES_T):
                    return BYTE_T
            return None
        if isinstance(expr, TSlice):
            obj_t = self.resolve(expr.obj)
            if obj_t is not None:
                if isinstance(obj_t, ListT):
                    return obj_t
                if type_eq(obj_t, STRING_T):
                    return STRING_T
                if type_eq(obj_t, BYTES_T):
                    return BYTES_T
            return None
        return None

    def resolve_call(self, expr: TCall) -> Type | None:
        if isinstance(expr.func, TVar):
            name = expr.func.name
            if name in self.checker.functions:
                return self.checker.functions[name].ret
            if name in self.checker.types:
                return self.checker.types[name]
            return self.resolve_builtin_call(name, expr)
        if isinstance(expr.func, TFieldAccess):
            obj_t = self.resolve(expr.func.obj)
            if isinstance(obj_t, StructT):
                if expr.func.field in obj_t.methods:
                    return obj_t.methods[expr.func.field].ret
        return None

    def resolve_builtin_call(self, name: str, expr: TCall) -> Type | None:
        return None

    def resolve_for_binder_types(self, stmt: TForStmt) -> dict[str, Type] | None:
        if isinstance(stmt.iterable, TRange):
            result: dict[str, Type] = {}
            for b in stmt.binding:
                result[b] = INT_T
            return result
        iter_type = self.resolve(stmt.iterable)
        if iter_type is None:
            return None
        result2: dict[str, Type] = {}
        if isinstance(iter_type, ListT):
            if len(stmt.binding) == 1:
                result2[stmt.binding[0]] = iter_type.element
            elif len(stmt.binding) == 2:
                result2[stmt.binding[0]] = INT_T
                result2[stmt.binding[1]] = iter_type.element
        elif type_eq(iter_type, STRING_T):
            if len(stmt.binding) == 1:
                result2[stmt.binding[0]] = RUNE_T
            elif len(stmt.binding) == 2:
                result2[stmt.binding[0]] = INT_T
                result2[stmt.binding[1]] = RUNE_T
        elif type_eq(iter_type, BYTES_T):
            if len(stmt.binding) == 1:
                result2[stmt.binding[0]] = BYTE_T
            elif len(stmt.binding) == 2:
                result2[stmt.binding[0]] = INT_T
                result2[stmt.binding[1]] = BYTE_T
        elif isinstance(iter_type, MapT):
            if len(stmt.binding) == 1:
                result2[stmt.binding[0]] = iter_type.key
            elif len(stmt.binding) == 2:
                result2[stmt.binding[0]] = iter_type.key
                result2[stmt.binding[1]] = iter_type.value
        elif isinstance(iter_type, SetT):
            if len(stmt.binding) == 1:
                result2[stmt.binding[0]] = iter_type.element
        else:
            return None
        return result2 if result2 else None


class ScopeResolver(TypeResolver):
    """Type resolver for scope analysis — handles scope-specific builtins."""

    def resolve_builtin_call(self, name: str, expr: TCall) -> Type | None:
        if name == "Len":
            return INT_T
        if name in (
            "Append",
            "Insert",
            "RemoveAt",
            "Delete",
            "Add",
            "Remove",
            "ReplaceSlice",
        ):
            return VOID_T
        if name == "Pop":
            if expr.args:
                t = self.resolve(expr.args[0].value)
                if t is not None and isinstance(t, ListT):
                    return t.element
            return None
        if name in ("FloorDiv", "PythonMod"):
            if expr.args:
                return self.resolve(expr.args[0].value)
            return INT_T
        if name == "ToString":
            return STRING_T
        if name in ("Keys", "Values"):
            if expr.args:
                t = self.resolve(expr.args[0].value)
                if t is not None and isinstance(t, MapT):
                    if name == "Keys":
                        return ListT(kind="list", element=t.key)
                    return ListT(kind="list", element=t.value)
            return None
        if name in ("Sorted", "Reversed"):
            if expr.args:
                return self.resolve(expr.args[0].value)
            return None
        if name in (
            "Concat",
            "Upper",
            "Lower",
            "Join",
            "Replace",
            "Trim",
            "TrimStart",
            "TrimEnd",
        ):
            return STRING_T
        if name in ("Split", "SplitN", "SplitWhitespace"):
            return ListT(kind="list", element=STRING_T)
        if name == "Args":
            return ListT(kind="list", element=STRING_T)
        return None


class StringsResolver(TypeResolver):
    """Type resolver for strings analysis — handles strings-specific builtins."""

    def resolve_builtin_call(self, name: str, expr: TCall) -> Type | None:
        if name in (
            "ToString",
            "Concat",
            "FormatInt",
            "Lower",
            "Upper",
            "Trim",
            "TrimStart",
            "TrimEnd",
            "Replace",
            "Repeat",
            "Reverse",
            "Join",
            "Format",
            "ReadAll",
            "Decode",
        ):
            return STRING_T
        if name == "ReadLine":
            return normalize_union([STRING_T, NIL_T])
        if name in ("Split", "SplitN", "SplitWhitespace"):
            return ListT(kind="list", element=STRING_T)
        if name == "Len":
            return INT_T
        if name == "Args":
            return ListT(kind="list", element=STRING_T)
        if name == "Keys":
            if expr.args:
                t = self.resolve(expr.args[0].value)
                if isinstance(t, MapT):
                    return ListT(kind="list", element=t.key)
        if name == "Values":
            if expr.args:
                t = self.resolve(expr.args[0].value)
                if isinstance(t, MapT):
                    return ListT(kind="list", element=t.value)
        if name == "Items":
            if expr.args:
                t = self.resolve(expr.args[0].value)
                if isinstance(t, MapT):
                    tup = TupleT(kind="tuple", elements=[t.key, t.value])
                    return ListT(kind="list", element=tup)
        if name == "Get":
            if expr.args:
                t = self.resolve(expr.args[0].value)
                if isinstance(t, MapT):
                    return normalize_union([t.value, NIL_T])
        if name == "Map":
            return None
        if name == "Set":
            return None
        return None
