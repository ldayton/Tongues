"""Taytsh typechecker — validates a parsed TModule against the spec's type rules."""

from __future__ import annotations

from dataclasses import dataclass

from .ast import (
    Pos,
    TArg,
    TAssignStmt,
    TBinaryOp,
    TBoolLit,
    TBreakStmt,
    TByteLit,
    TBytesLit,
    TCall,
    TContinueStmt,
    TEnumDecl,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFnDecl,
    TFnLit,
    TFloatLit,
    TForStmt,
    TFuncType,
    TIdentType,
    TIfStmt,
    TIndex,
    TIntLit,
    TInterfaceDecl,
    TLetStmt,
    TListLit,
    TListType,
    TMapLit,
    TMapType,
    TMatchCase,
    TMatchStmt,
    TModule,
    TNilLit,
    TOpAssignStmt,
    TOptionalType,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TPrimitive,
    TRange,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TSetType,
    TSlice,
    TStmt,
    TStringLit,
    TStructDecl,
    TThrowStmt,
    TTernary,
    TTupleAccess,
    TTupleAssignStmt,
    TTupleLit,
    TTupleType,
    TTryStmt,
    TType,
    TUnaryOp,
    TUnionType,
    TVar,
    TWhileStmt,
)


# ============================================================
# RESOLVED TYPE REPRESENTATION
# ============================================================

TY_INT: str = "int"
TY_FLOAT: str = "float"
TY_BOOL: str = "bool"
TY_BYTE: str = "byte"
TY_BYTES: str = "bytes"
TY_STRING: str = "string"
TY_RUNE: str = "rune"
TY_NIL: str = "nil"
TY_VOID: str = "void"
TY_ERROR: str = "error"


@dataclass
class Type:
    kind: str


@dataclass
class ListT(Type):
    element: Type


@dataclass
class MapT(Type):
    key: Type
    value: Type


@dataclass
class SetT(Type):
    element: Type


@dataclass
class TupleT(Type):
    elements: list[Type]


@dataclass
class FnT(Type):
    params: list[Type]
    ret: Type
    min_params: int = -1


@dataclass
class StructT(Type):
    name: str
    fields: dict[str, Type]
    methods: dict[str, FnT]
    parent: str | None
    field_order: list[str]
    min_fields: int = -1


@dataclass
class InterfaceT(Type):
    name: str
    variants: list[str]


@dataclass
class EnumT(Type):
    name: str
    variants: list[str]


@dataclass
class UnionT(Type):
    members: list[Type]


# Primitive singletons
INT_T: Type = Type(kind=TY_INT)
FLOAT_T: Type = Type(kind=TY_FLOAT)
BOOL_T: Type = Type(kind=TY_BOOL)
BYTE_T: Type = Type(kind=TY_BYTE)
BYTES_T: Type = Type(kind=TY_BYTES)
STRING_T: Type = Type(kind=TY_STRING)
RUNE_T: Type = Type(kind=TY_RUNE)
NIL_T: Type = Type(kind=TY_NIL)
VOID_T: Type = Type(kind=TY_VOID)
ERROR_T: Type = Type(kind=TY_ERROR)

_PRIMITIVE_MAP: dict[str, Type] = {
    "int": INT_T,
    "float": FLOAT_T,
    "bool": BOOL_T,
    "byte": BYTE_T,
    "bytes": BYTES_T,
    "string": STRING_T,
    "rune": RUNE_T,
    "nil": NIL_T,
    "void": VOID_T,
    "error": ERROR_T,
}


# ============================================================
# TYPE EQUALITY
# ============================================================


def type_eq(a: Type, b: Type) -> bool:
    if a.kind != b.kind:
        return False
    if isinstance(a, ListT) and isinstance(b, ListT):
        return type_eq(a.element, b.element)
    if isinstance(a, MapT) and isinstance(b, MapT):
        return type_eq(a.key, b.key) and type_eq(a.value, b.value)
    if isinstance(a, SetT) and isinstance(b, SetT):
        return type_eq(a.element, b.element)
    if isinstance(a, TupleT) and isinstance(b, TupleT):
        if len(a.elements) != len(b.elements):
            return False
        i = 0
        while i < len(a.elements):
            if not type_eq(a.elements[i], b.elements[i]):
                return False
            i += 1
        return True
    if isinstance(a, FnT) and isinstance(b, FnT):
        if len(a.params) != len(b.params):
            return False
        i = 0
        while i < len(a.params):
            if not type_eq(a.params[i], b.params[i]):
                return False
            i += 1
        return type_eq(a.ret, b.ret)
    if isinstance(a, StructT) and isinstance(b, StructT):
        return a.name == b.name
    if isinstance(a, InterfaceT) and isinstance(b, InterfaceT):
        return a.name == b.name
    if isinstance(a, EnumT) and isinstance(b, EnumT):
        return a.name == b.name
    if isinstance(a, UnionT) and isinstance(b, UnionT):
        return _union_members_eq(a.members, b.members)
    return a.kind == b.kind


def _container_eq(a: Type, b: Type) -> bool:
    """type_eq but error/void (empty literals) match anything recursively."""
    if a.kind == TY_ERROR or a.kind == TY_VOID:
        return True
    if b.kind == TY_ERROR or b.kind == TY_VOID:
        return True
    if isinstance(a, ListT) and isinstance(b, ListT):
        return _container_eq(a.element, b.element)
    if isinstance(a, MapT) and isinstance(b, MapT):
        return _container_eq(a.key, b.key) and _container_eq(a.value, b.value)
    if isinstance(a, SetT) and isinstance(b, SetT):
        return _container_eq(a.element, b.element)
    return type_eq(a, b)


def _union_members_eq(a: list[Type], b: list[Type]) -> bool:
    if len(a) != len(b):
        return False
    for m in a:
        found = False
        for n in b:
            if type_eq(m, n):
                found = True
                break
        if not found:
            return False
    return True


# ============================================================
# TYPE KEYS
# ============================================================


def _type_key(t: Type) -> str:
    """Produce a string key for a resolved Type, for use in sets."""
    if isinstance(t, ListT):
        return "list[" + _type_key(t.element) + "]"
    if isinstance(t, MapT):
        return "map[" + _type_key(t.key) + "," + _type_key(t.value) + "]"
    if isinstance(t, SetT):
        return "set[" + _type_key(t.element) + "]"
    if isinstance(t, TupleT):
        parts: list[str] = []
        for e in t.elements:
            parts.append(_type_key(e))
        return "(" + ",".join(parts) + ")"
    if isinstance(t, FnT):
        parts2: list[str] = []
        for p in t.params:
            parts2.append(_type_key(p))
        return "fn[" + ",".join(parts2) + "->" + _type_key(t.ret) + "]"
    if isinstance(t, StructT):
        return "struct:" + t.name
    if isinstance(t, InterfaceT):
        return "interface:" + t.name
    if isinstance(t, EnumT):
        return "enum:" + t.name
    if isinstance(t, UnionT):
        keys: list[str] = []
        for m in t.members:
            keys.append(_type_key(m))
        keys.sort()
        return "union{" + "|".join(keys) + "}"
    return t.kind


def type_name(t: Type) -> str:
    """Human-readable name for a type, for error messages."""
    if isinstance(t, ListT):
        return "list[" + type_name(t.element) + "]"
    if isinstance(t, MapT):
        return "map[" + type_name(t.key) + ", " + type_name(t.value) + "]"
    if isinstance(t, SetT):
        return "set[" + type_name(t.element) + "]"
    if isinstance(t, TupleT):
        parts: list[str] = []
        for e in t.elements:
            parts.append(type_name(e))
        return "(" + ", ".join(parts) + ")"
    if isinstance(t, FnT):
        parts2: list[str] = []
        for p in t.params:
            parts2.append(type_name(p))
        parts2.append(type_name(t.ret))
        return "fn[" + ", ".join(parts2) + "]"
    if isinstance(t, StructT):
        return t.name
    if isinstance(t, InterfaceT):
        return t.name
    if isinstance(t, EnumT):
        return t.name
    if isinstance(t, UnionT):
        keyed: list[tuple[str, Type]] = []
        for m in t.members:
            keyed.append((_type_key(m), m))
        # Insertion sort by string key (tuples not sortable in Taytsh)
        si = 1
        while si < len(keyed):
            skey = keyed[si]
            sj = si - 1
            while sj >= 0 and keyed[sj][0] > skey[0]:
                keyed[sj + 1] = keyed[sj]
                sj -= 1
            keyed[sj + 1] = skey
            si += 1
        parts3: list[str] = []
        for _k, m in keyed:
            parts3.append(type_name(m))
        return " | ".join(parts3)
    return t.kind


# ============================================================
# UNION NORMALIZATION
# ============================================================


def normalize_union(members: list[Type]) -> Type:
    flat: list[Type] = []
    for m in members:
        if isinstance(m, UnionT):
            for inner in m.members:
                flat.append(inner)
        else:
            flat.append(m)
    # Deduplicate
    deduped: list[Type] = []
    for m in flat:
        found = False
        for existing in deduped:
            if type_eq(m, existing):
                found = True
                break
        if not found:
            deduped.append(m)
    # Absorb error
    for m in deduped:
        if m.kind == TY_ERROR:
            return ERROR_T
    if len(deduped) == 1:
        return deduped[0]
    return UnionT(kind="union", members=deduped)


def make_optional(inner: Type) -> Type:
    """Make T? — union of inner and nil."""
    if type_eq(inner, NIL_T):
        return NIL_T
    if isinstance(inner, UnionT):
        has_nil = False
        for m in inner.members:
            if type_eq(m, NIL_T):
                has_nil = True
                break
        if has_nil:
            return inner
        return normalize_union(list(inner.members) + [NIL_T])
    return normalize_union([inner, NIL_T])


def contains_nil(t: Type) -> bool:
    if type_eq(t, NIL_T):
        return True
    if isinstance(t, UnionT):
        for m in t.members:
            if type_eq(m, NIL_T):
                return True
    return False


def remove_nil(t: Type) -> Type:
    """Remove nil from a type (for narrowing)."""
    if type_eq(t, NIL_T):
        return NIL_T
    if isinstance(t, UnionT):
        remaining: list[Type] = []
        for m in t.members:
            if not type_eq(m, NIL_T):
                remaining.append(m)
        if len(remaining) == 0:
            return NIL_T
        if len(remaining) == 1:
            return remaining[0]
        return UnionT(kind="union", members=remaining)
    return t


def _unwrap_nil_union(t: Type) -> Type:
    """If t is a union containing nil, return t with nil removed. Otherwise return t."""
    if isinstance(t, UnionT):
        non_nil = remove_nil(t)
        if not type_eq(non_nil, t):
            return non_nil
    return t


def _literal_key_value(expr: TExpr) -> str | None:
    """Extract a comparable string key from a literal key expression."""
    if isinstance(expr, TIntLit):
        return "int:" + str(expr.value)
    if isinstance(expr, TStringLit):
        return "string:" + expr.value
    if isinstance(expr, TBoolLit):
        return "bool:" + str(expr.value)
    if isinstance(expr, TFloatLit):
        return "float:" + expr.raw
    return None


def _contains_fn_type(t: Type) -> bool:
    """Return True if the type is or contains a fn type."""
    if isinstance(t, FnT):
        return True
    if isinstance(t, ListT):
        return _contains_fn_type(t.element)
    if isinstance(t, MapT):
        return _contains_fn_type(t.key) or _contains_fn_type(t.value)
    if isinstance(t, SetT):
        return _contains_fn_type(t.element)
    if isinstance(t, TupleT):
        for elem in t.elements:
            if _contains_fn_type(elem):
                return True
    if isinstance(t, UnionT):
        for m in t.members:
            if _contains_fn_type(m):
                return True
    return False


def _has_zero_value(t: Type) -> bool:
    """Return True if the type has a zero/default value (no initializer needed)."""
    if t.kind in (TY_INT, TY_FLOAT, TY_BOOL, TY_BYTE, TY_BYTES, TY_STRING, TY_RUNE):
        return True
    if t.kind == TY_NIL:
        return True
    if isinstance(t, ListT):
        return True
    if isinstance(t, MapT):
        return True
    if isinstance(t, SetT):
        return True
    if isinstance(t, TupleT):
        for elem in t.elements:
            if not _has_zero_value(elem):
                return False
        return True
    if isinstance(t, UnionT):
        return contains_nil(t)
    if isinstance(t, FnT):
        return type_eq(t.ret, VOID_T) and len(t.params) == 0
    return False


def _block_is_complete(stmts: list[TStmt]) -> bool:
    """Return True if all paths through this block return or throw."""
    if len(stmts) == 0:
        return False
    last = stmts[len(stmts) - 1]
    if isinstance(last, (TReturnStmt, TThrowStmt)):
        return True
    if isinstance(last, TExprStmt) and isinstance(last.expr, TCall):
        if isinstance(last.expr.func, TVar) and last.expr.func.name == "Exit":
            return True
    if isinstance(last, TIfStmt):
        if last.else_body is None:
            return False
        return _block_is_complete(last.then_body) and _block_is_complete(last.else_body)
    if isinstance(last, TWhileStmt):
        if isinstance(last.cond, TBoolLit) and last.cond.value is True:
            return not _stmts_contain_break(last.body)
        return False
    if isinstance(last, TMatchStmt):
        if last.default is not None:
            for case in last.cases:
                if not _block_is_complete(case.body):
                    return False
            return _block_is_complete(last.default.body)
        if len(last.cases) > 0:
            for case in last.cases:
                if not _block_is_complete(case.body):
                    return False
            return True
        return False
    if isinstance(last, TTryStmt):
        if len(last.catches) == 0:
            return _block_is_complete(last.body)
        return _block_is_complete(last.body) or all(
            _block_is_complete(c.body) for c in last.catches
        )
    return False


def _block_always_exits(stmts: list[TStmt]) -> bool:
    """Return True if all paths exit via return, throw, break, or continue."""
    if len(stmts) == 0:
        return False
    last = stmts[len(stmts) - 1]
    if isinstance(last, (TReturnStmt, TThrowStmt, TBreakStmt, TContinueStmt)):
        return True
    if isinstance(last, TExprStmt) and isinstance(last.expr, TCall):
        if isinstance(last.expr.func, TVar) and last.expr.func.name == "Exit":
            return True
    if isinstance(last, TIfStmt):
        if last.else_body is None:
            return False
        return _block_always_exits(last.then_body) and _block_always_exits(
            last.else_body
        )
    if isinstance(last, TWhileStmt):
        if isinstance(last.cond, TBoolLit) and last.cond.value is True:
            return not _stmts_contain_break(last.body)
        return False
    if isinstance(last, TMatchStmt):
        if last.default is not None:
            for case in last.cases:
                if not _block_always_exits(case.body):
                    return False
            return _block_always_exits(last.default.body)
        if len(last.cases) > 0:
            for case in last.cases:
                if not _block_always_exits(case.body):
                    return False
            return True
        return False
    if isinstance(last, TTryStmt):
        if len(last.catches) == 0:
            return _block_always_exits(last.body)
        return _block_always_exits(last.body) or all(
            _block_always_exits(c.body) for c in last.catches
        )
    return False


def _stmts_contain_break(stmts: list[TStmt]) -> bool:
    """Return True if any statement in the list is or contains a break."""
    for s in stmts:
        if isinstance(s, TBreakStmt):
            return True
        if isinstance(s, TIfStmt):
            if _stmts_contain_break(s.then_body):
                return True
            if s.else_body is not None and _stmts_contain_break(s.else_body):
                return True
        if isinstance(s, TTryStmt):
            if _stmts_contain_break(s.body):
                return True
            for c in s.catches:
                if _stmts_contain_break(c.body):
                    return True
        if isinstance(s, TMatchStmt):
            for c in s.cases:
                if _stmts_contain_break(c.body):
                    return True
            if s.default is not None and _stmts_contain_break(s.default.body):
                return True
    return False


def _body_always_exits(stmts: list[TStmt]) -> bool:
    """Return True if the last statement unconditionally exits the block."""
    if len(stmts) == 0:
        return False
    last = stmts[len(stmts) - 1]
    return isinstance(last, (TReturnStmt, TBreakStmt, TContinueStmt, TThrowStmt))


def _field_access_path(expr: TExpr) -> str | None:
    """Build a dotted path from nested TFieldAccess, e.g. 'a.b.c'."""
    if isinstance(expr, TVar):
        return expr.name
    if isinstance(expr, TFieldAccess):
        base = _field_access_path(expr.obj)
        if base is not None:
            return base + "." + expr.field
    return None


def _nil_check_var(cond: TExpr) -> tuple[str, str] | None:
    """Extract (var_name, "is_nil"|"is_not_nil") from a nil-check condition."""
    # x == nil / x != nil / a.b.c == nil / a.b.c != nil
    if isinstance(cond, TBinaryOp) and isinstance(cond.right, TNilLit):
        path = _field_access_path(cond.left)
        if path is not None:
            if cond.op == "==":
                return (path, "is_nil")
            if cond.op == "!=":
                return (path, "is_not_nil")
    # IsNil(x) / IsNil(a.b.c)
    if (
        isinstance(cond, TCall)
        and isinstance(cond.func, TVar)
        and cond.func.name == "IsNil"
        and len(cond.args) >= 1
    ):
        path = _field_access_path(cond.args[0].value)
        if path is not None:
            return (path, "is_nil")
    # !IsNil(x) / !IsNil(a.b.c)
    if isinstance(cond, TUnaryOp) and cond.op == "!":
        inner = cond.operand
        if (
            isinstance(inner, TCall)
            and isinstance(inner.func, TVar)
            and inner.func.name == "IsNil"
            and len(inner.args) >= 1
        ):
            path = _field_access_path(inner.args[0].value)
            if path is not None:
                return (path, "is_not_nil")
    return None


def _collect_nil_checks(
    cond: TExpr, binary_only: bool, chain_op: str
) -> list[tuple[str, str]]:
    """Extract nil-check variables from a condition, walking into chains."""
    result: list[tuple[str, str]] = []
    if isinstance(cond, TBinaryOp) and cond.op == chain_op:
        result.extend(_collect_nil_checks(cond.left, binary_only, chain_op))
        result.extend(_collect_nil_checks(cond.right, binary_only, chain_op))
        return result
    if binary_only:
        if isinstance(cond, TBinaryOp) and isinstance(cond.right, TNilLit):
            path = _field_access_path(cond.left)
            if path is not None:
                if cond.op == "==":
                    result.append((path, "is_nil"))
                elif cond.op == "!=":
                    result.append((path, "is_not_nil"))
    else:
        info = _nil_check_var(cond)
        if info is not None:
            result.append(info)
    return result


def _istype_var_from_call(call: TCall) -> tuple[str, str] | None:
    """Extract (var_path, type_name) from IsType(x, "T")."""
    if not (isinstance(call.func, TVar) and call.func.name == "IsType"):
        return None
    if len(call.args) < 2:
        return None
    first = call.args[0].value
    second = call.args[1].value
    if not isinstance(second, TStringLit):
        return None
    type_name = second.value
    path = _field_access_path(first)
    if path is not None:
        return (path, type_name)
    return None


def _istype_var_from_call_pos(
    call: TCall, positive: bool
) -> tuple[str, str, bool] | None:
    """Extract (var_path, type_name, is_positive) from IsType(x, "T")."""
    if not (isinstance(call.func, TVar) and call.func.name == "IsType"):
        return None
    if len(call.args) < 2:
        return None
    first = call.args[0].value
    second = call.args[1].value
    if not isinstance(second, TStringLit):
        return None
    type_name = second.value
    path = _field_access_path(first)
    if path is not None:
        return (path, type_name, positive)
    return None


def _type_check_var(cond: TExpr) -> tuple[str, str, bool] | None:
    """Extract (var_path, type_name, is_positive) from a type-check condition."""
    # IsType(x, "T")
    if isinstance(cond, TCall):
        return _istype_var_from_call_pos(cond, True)
    # !IsType(x, "T")
    if isinstance(cond, TUnaryOp) and cond.op == "!":
        inner = cond.operand
        if isinstance(inner, TCall):
            return _istype_var_from_call_pos(inner, False)
    return None


def _collect_type_checks(cond: TExpr) -> list[tuple[str, str, bool]]:
    """Extract type checks from a condition, walking && and || chains."""
    result: list[tuple[str, str, bool]] = []
    # && chains: each arm narrows independently
    if isinstance(cond, TBinaryOp) and cond.op == "&&":
        result.extend(_collect_type_checks(cond.left))
        result.extend(_collect_type_checks(cond.right))
        return result
    # || chains: can't narrow to a single type, skip
    if isinstance(cond, TBinaryOp) and cond.op == "||":
        return []
    single = _type_check_var(cond)
    if single is not None:
        result.append(single)
    return result


def _collect_type_checks_guard(cond: TExpr) -> list[tuple[str, str, bool]]:
    """Extract type checks for guard narrowing (after early exit).
    Only walks || chains (sound under negation). Does NOT recurse into &&."""
    result: list[tuple[str, str, bool]] = []
    if isinstance(cond, TBinaryOp) and cond.op == "||":
        result.extend(_collect_type_checks_guard(cond.left))
        result.extend(_collect_type_checks_guard(cond.right))
        return result
    if isinstance(cond, TBinaryOp) and cond.op == "&&":
        return []
    single = _type_check_var(cond)
    if single is not None:
        result.append(single)
    return result


def _collect_nil_checks_guard(cond: TExpr) -> list[tuple[str, str]]:
    """Extract nil checks for guard narrowing (after early exit).
    Only walks || chains (sound under negation). Does NOT recurse into &&."""
    result: list[tuple[str, str]] = []
    if isinstance(cond, TBinaryOp) and cond.op == "||":
        result.extend(_collect_nil_checks_guard(cond.left))
        result.extend(_collect_nil_checks_guard(cond.right))
        return result
    if isinstance(cond, TBinaryOp) and cond.op == "&&":
        return []
    info = _nil_check_var(cond)
    if info is not None:
        result.append(info)
    return result


def _is_truthy_type(t: Type) -> bool:
    """Return True if a type supports truthiness (can be used as a bool condition)."""
    return False


# ============================================================
# ASSIGNABILITY
# ============================================================


def is_assignable(source: Type, target: Type) -> bool:
    """Can a value of type `source` be assigned to a slot of type `target`?"""
    if type_eq(source, target):
        return True
    # Source is struct or interface, target is interface containing it
    if isinstance(target, InterfaceT):
        if isinstance(source, (StructT, InterfaceT)) and source.name in target.variants:
            return True
        if isinstance(source, MapT) and is_assignable(source.value, target):
            return True
    # Error type: assignable to/from anything (prevents cascading)
    if source.kind == TY_ERROR or target.kind == TY_ERROR:
        return True
    # Source is nil, target contains nil
    if source.kind == TY_NIL and contains_nil(target):
        return True
    # Target is union, source is a member
    if isinstance(target, UnionT):
        for m in target.members:
            if is_assignable(source, m):
                return True
    # Source is union, all members assignable to target
    if isinstance(source, UnionT):
        all_ok = True
        for m in source.members:
            if not is_assignable(m, target):
                all_ok = False
                break
        if all_ok:
            return True
    # Containers are invariant, but error/void elements (empty literals) match anything
    if isinstance(source, ListT) and isinstance(target, ListT):
        if source.element.kind == TY_ERROR or source.element.kind == TY_VOID:
            return True
        if _container_eq(source.element, target.element):
            return True
    if isinstance(source, MapT) and isinstance(target, MapT):
        if source.key.kind == TY_ERROR and source.value.kind == TY_ERROR:
            return True
        if source.key.kind == TY_VOID:
            return True
        if _container_eq(source.key, target.key) and _container_eq(
            source.value, target.value
        ):
            return True
    if isinstance(source, SetT) and isinstance(target, SetT):
        if source.element.kind == TY_ERROR or source.element.kind == TY_VOID:
            return True
        if _container_eq(source.element, target.element):
            return True
    # Tuple element-by-element assignability
    if isinstance(source, TupleT) and isinstance(target, TupleT):
        if len(source.elements) == len(target.elements):
            ok = True
            for i in range(len(source.elements)):
                if not is_assignable(source.elements[i], target.elements[i]):
                    ok = False
                    break
            if ok:
                return True
    # list[T] assignable to tuple of T elements (tuple augmented assignment)
    if isinstance(source, ListT) and isinstance(target, TupleT):
        if len(target.elements) > 0:
            all_ok = True
            ei = 0
            while ei < len(target.elements):
                if not is_assignable(source.element, target.elements[ei]):
                    all_ok = False
                    break
                ei += 1
            if all_ok:
                return True
    return False


# ============================================================
# ZERO VALUES
# ============================================================


def is_hashable(t: Type) -> bool:
    """Check if a type is hashable (valid as map key or set element)."""
    if t.kind in (
        TY_INT,
        TY_FLOAT,
        TY_BOOL,
        TY_BYTE,
        TY_BYTES,
        TY_STRING,
        TY_RUNE,
        TY_NIL,
    ):
        return True
    if isinstance(t, EnumT):
        return True
    if isinstance(t, TupleT):
        for e in t.elements:
            if not is_hashable(e):
                return False
        return True
    return False


# ============================================================
# BUILT-IN NAMES (reserved)
# ============================================================

BUILTIN_NAMES: set[str] = {
    # Numeric
    "Abs",
    "Min",
    "Max",
    "Sum",
    "Pow",
    "Round",
    "Floor",
    "Ceil",
    "DivMod",
    # Bytes
    "Encode",
    "Decode",
    # Strings
    "Len",
    "Concat",
    "RuneFromInt",
    "RuneToInt",
    "ParseInt",
    "ParseFloat",
    "FormatInt",
    "Upper",
    "Lower",
    "Trim",
    "TrimStart",
    "TrimEnd",
    "Split",
    "SplitN",
    "SplitWhitespace",
    "Join",
    "Find",
    "RFind",
    "Count",
    "Contains",
    "Replace",
    "ReplaceCount",
    "Repeat",
    "Reverse",
    "StartsWith",
    "EndsWith",
    "IsDigit",
    "IsAlpha",
    "IsAlnum",
    "IsSpace",
    "IsUpper",
    "IsLower",
    "Format",
    # Lists
    "Append",
    "Insert",
    "Pop",
    "RemoveAt",
    "IndexOf",
    "Reversed",
    "Sorted",
    # Maps
    "Map",
    "Get",
    "Delete",
    "Keys",
    "Values",
    "Items",
    "Merge",
    "PopItem",
    "MapFromKeys",
    # Sets
    "Set",
    "Add",
    "Remove",
    "Union",
    "Intersection",
    "Difference",
    # Conversions
    "IntToFloat",
    "FloatToInt",
    "ByteToInt",
    "IntToByte",
    "ToString",
    # I/O
    "WriteOut",
    "WriteErr",
    "WritelnOut",
    "WritelnErr",
    "ReadLine",
    "ReadAll",
    "ReadBytes",
    "ReadBytesN",
    "ReadFile",
    "ReadFileBytes",
    "WriteFile",
    "Args",
    "GetEnv",
    "Exit",
    # Assert / Unwrap
    "Assert",
    "Unwrap",
    # Math
    "IsNaN",
    "IsInf",
    "Sqrt",
    # Arithmetic
    "FloorDiv",
    "PythonMod",
    "WrappingAdd",
    "WrappingSub",
    "WrappingMul",
    # Bytes constructors
    "Bytes",
    "BytesFrom",
    # Range-to-list
    "RangeList",
    # Map constructor
    "MapFromPairs",
    # List comparison
    "ListCompare",
    # Zip
    "Zip",
    # Set from iterable
    "SetFromList",
    # String to char list
    "Chars",
    # List from iterable
    "ListFrom",
    # Mutation
    "ReplaceSlice",
    # Nil check
    "IsNil",
    # Type check
    "IsType",
}


# Names reserved for user bindings (top-level decls, locals, params, etc.).
# Most builtins are reserved; set-specific operations like Add can be shadowed.
RESERVED_NAMES: set[str] = set(BUILTIN_NAMES) - {"Add"}

# Built-in error struct names
BUILTIN_STRUCTS: dict[str, dict[str, Type]] = {
    "KeyError": {"message": STRING_T},
    "IndexError": {"message": STRING_T},
    "ZeroDivisionError": {"message": STRING_T},
    "AssertError": {"message": STRING_T},
    "NilError": {"message": STRING_T},
    "ValueError": {"message": STRING_T},
    "TypeError": {"message": STRING_T},
    "NotImplementedError": {"message": STRING_T},
    "RuntimeError": {"message": STRING_T},
    "IOError": {"message": STRING_T},
    "Exception": {"message": STRING_T},
    "BaseException": {"message": STRING_T},
}


# ============================================================
# CHECK ERROR
# ============================================================


class CheckError:
    """An error found during type checking."""

    def __init__(self, msg: str, line: int, col: int, source_file: str = "") -> None:
        self.msg: str = msg
        self.line: int = line
        self.col: int = col
        self.source_file: str = source_file

    def __repr__(self) -> str:
        file_prefix = ""
        if self.source_file != "":
            file_prefix = self.source_file + ":"
        return (
            file_prefix
            + "error:"
            + str(self.line)
            + ":"
            + str(self.col)
            + ": [check] "
            + self.msg
        )


class _BuiltinCtx:
    """Helper for check_builtin_call — holds call context."""

    checker: Checker
    name: str
    arg_types: list[Type | None]
    n: int
    pos: Pos

    def __init__(
        self,
        checker: Checker,
        name: str,
        arg_types: list[Type | None],
        n: int,
        pos: Pos,
    ) -> None:
        self.checker = checker
        self.name = name
        self.arg_types = arg_types
        self.n = n
        self.pos = pos


def _bctx_require(ctx: _BuiltinCtx, count: int) -> bool:
    if ctx.n != count:
        ctx.checker.error(
            ctx.name + " requires " + str(count) + " argument(s), got " + str(ctx.n),
            ctx.pos,
        )
        return False
    return True


def _bctx_require_range(ctx: _BuiltinCtx, lo: int, hi: int) -> bool:
    if ctx.n < lo or ctx.n > hi:
        ctx.checker.error(
            ctx.name
            + " requires "
            + str(lo)
            + "-"
            + str(hi)
            + " argument(s), got "
            + str(ctx.n),
            ctx.pos,
        )
        return False
    return True


def _bctx_arg(ctx: _BuiltinCtx, i: int) -> Type | None:
    if i < len(ctx.arg_types):
        return ctx.arg_types[i]
    return None


# ============================================================
# CHECKER
# ============================================================


class Checker:
    def __init__(self) -> None:
        self.errors: list[CheckError] = []
        self.types: dict[str, Type] = {}
        self.functions: dict[str, FnT] = {}
        self.fn_param_names: dict[str, list[str]] = {}
        self.scopes: list[dict[str, Type]] = []
        self.current_fn_ret: Type | None = None
        self.in_loop: bool = False
        self.current_struct: StructT | None = None
        self.strict_math: bool = False
        self.loop_vars: set[str] = set()
        self.in_finally: bool = False
        self.uninitialized: set[str] = set()
        self._declared: dict[str, Type] = {}
        self.expr_types: dict[tuple[int, int], Type] = {}
        self.bool_facts: dict[str, TExpr] = {}

    def error(self, msg: str, pos: Pos) -> None:
        self.errors.append(CheckError(msg, pos.line, pos.col, pos.source_file))

    def _resolve_bool_var(self, cond: TExpr) -> TExpr:
        """If cond is a bool variable with a stored narrowing fact, return it."""
        if isinstance(cond, TVar) and cond.name in self.bool_facts:
            return self.bool_facts[cond.name]
        return cond

    # ── Scope management ──────────────────────────────────────

    def enter_scope(self) -> None:
        self.scopes.append({})

    def exit_scope(self) -> None:
        self.scopes.pop()

    def declare(self, name: str, typ: Type, pos: Pos) -> None:
        if name == "_":
            return
        if name in RESERVED_NAMES:
            self.error("cannot use reserved name '" + name + "'", pos)
            return
        # Check current scope for duplicate
        if len(self.scopes) > 0 and name in self.scopes[-1]:
            self.error("'" + name + "' shadows outer binding", pos)
            return
        # Check outer scopes for shadowing
        i = len(self.scopes) - 2
        while i >= 0:
            if name in self.scopes[i]:
                self.error("'" + name + "' shadows outer binding", pos)
                return
            i -= 1
        if len(self.scopes) > 0:
            self.scopes[-1][name] = typ
            self._declared[name] = typ

    def narrow(self, name: str, typ: Type) -> None:
        """Shadow an outer binding with a narrowed type in the current scope."""
        if len(self.scopes) > 0:
            self.scopes[-1][name] = typ

    def _try_lookup(self, name: str) -> Type | None:
        """Look up a name without emitting errors."""
        i = len(self.scopes) - 1
        while i >= 0:
            if name in self.scopes[i]:
                return self.scopes[i][name]
            i -= 1
        if name in self.functions:
            return self.functions[name]
        if name in self.types:
            return self.types[name]
        return None

    def lookup(self, name: str, pos: Pos) -> Type | None:
        r = self._try_lookup(name)
        if r is not None:
            return r
        self.error("undefined name '" + name + "'", pos)
        return None

    def lookup_declared(self, name: str, pos: Pos) -> Type | None:
        """Look up the declared (not narrowed) type of a variable."""
        if name in self._declared:
            return self._declared[name]
        return self.lookup(name, pos)

    def _lookup_field_type(self, path: str, pos: Pos) -> Type | None:
        """Look up the type of a dotted field path like 'a.b' or 'a.b.c'."""
        parts = path.split(".")
        if len(parts) < 2:
            return None
        # Check if a prefix is narrowed
        current = self.lookup(parts[0], pos)
        if current is None:
            return None
        pi = 1
        while pi < len(parts):
            # Check if current prefix path is narrowed
            prefix = ".".join(parts[: pi + 1])
            narrowed = self._lookup_narrowed_path(prefix)
            if narrowed is not None:
                current = narrowed
                pi += 1
                continue
            if isinstance(current, StructT) and parts[pi] in current.fields:
                current = current.fields[parts[pi]]
            else:
                return None
            pi += 1
        return current

    def _lookup_narrowed_path(self, path: str) -> Type | None:
        """Look up a narrowed dotted path in scopes."""
        i = len(self.scopes) - 1
        while i >= 0:
            if path in self.scopes[i]:
                return self.scopes[i][path]
            i -= 1
        return None

    def _narrow_to_type(self, current_type: Type, target_name: str) -> Type | None:
        """Resolve an IsType check against current type, returning the narrowed type."""
        if isinstance(current_type, InterfaceT):
            if target_name in current_type.variants:
                return self.types.get(target_name)
        if isinstance(current_type, UnionT):
            for m in current_type.members:
                if isinstance(m, (StructT, InterfaceT)) and m.name == target_name:
                    return m
                if isinstance(m, InterfaceT) and target_name in m.variants:
                    return self.types.get(target_name)
        return None

    # ── Type resolution ───────────────────────────────────────

    def resolve_type(self, t: TType) -> Type:
        """Resolve a parse-time TType node into a checked Type."""
        if isinstance(t, TPrimitive):
            result = _PRIMITIVE_MAP.get(t.kind)
            if result is None:
                self.error("unknown primitive type '" + t.kind + "'", t.pos)
                return ERROR_T
            return result
        if isinstance(t, TListType):
            elem = self.resolve_type(t.element)
            if type_eq(elem, VOID_T):
                self.error("void is not a value type", t.pos)
                return ListT(kind="list", element=ERROR_T)
            return ListT(kind="list", element=elem)
        if isinstance(t, TMapType):
            key = self.resolve_type(t.key)
            value = self.resolve_type(t.value)
            if type_eq(key, VOID_T) or type_eq(value, VOID_T):
                self.error("void is not a value type", t.pos)
                return MapT(
                    kind="map",
                    key=ERROR_T if type_eq(key, VOID_T) else key,
                    value=ERROR_T if type_eq(value, VOID_T) else value,
                )
            if not type_eq(key, VOID_T) and not is_hashable(key):
                self.error(type_name(key) + " is not hashable", t.pos)
            return MapT(kind="map", key=key, value=value)
        if isinstance(t, TSetType):
            elem = self.resolve_type(t.element)
            if type_eq(elem, VOID_T):
                self.error("void is not a value type", t.pos)
                return SetT(kind="set", element=ERROR_T)
            elif not is_hashable(elem):
                self.error(type_name(elem) + " is not hashable", t.pos)
            return SetT(kind="set", element=elem)
        if isinstance(t, TTupleType):
            elems: list[Type] = []
            for e in t.elements:
                resolved = self.resolve_type(e)
                if type_eq(resolved, VOID_T):
                    self.error("void is not a value type", t.pos)
                elems.append(resolved)
            return TupleT(kind="tuple", elements=elems)
        if isinstance(t, TFuncType):
            if len(t.params) < 1:
                self.error("fn type must have at least a return type", t.pos)
                return ERROR_T
            params: list[Type] = []
            i = 0
            while i < len(t.params) - 1:
                p = self.resolve_type(t.params[i])
                if type_eq(p, VOID_T):
                    self.error("void is not a value type", t.params[i].pos)
                params.append(p)
                i += 1
            ret = self.resolve_type(t.params[-1])
            return FnT(kind="fn", params=params, ret=ret)
        if isinstance(t, TIdentType):
            if t.name in self.types:
                return self.types[t.name]
            return ERROR_T
        if isinstance(t, TUnionType):
            members: list[Type] = []
            for m in t.members:
                resolved = self.resolve_type(m)
                if type_eq(resolved, VOID_T):
                    self.error("void is not a value type", m.pos)
                members.append(resolved)
            return normalize_union(members)
        if isinstance(t, TOptionalType):
            inner = self.resolve_type(t.inner)
            if type_eq(inner, VOID_T):
                self.error("void cannot be used as optional base type", t.pos)
                return ERROR_T
            return make_optional(inner)
        self.error("unhandled type node", t.pos)
        return ERROR_T

    # ── Pass 1: Collect declarations ──────────────────────────

    def collect_declarations(self, module: TModule) -> None:
        # Register built-in error structs before user declarations
        for bname in BUILTIN_STRUCTS:
            bfields = BUILTIN_STRUCTS[bname]
            st = StructT(
                kind="struct",
                name=bname,
                fields=bfields,
                methods={},
                parent=None,
                field_order=list(bfields.keys()),
            )
            self.types[bname] = st
        # First pass: register all type names (structs, interfaces, enums)
        # so they can reference each other
        for decl in module.decls:
            if isinstance(decl, TStructDecl):
                if decl.name in self.types:
                    self.error("duplicate type name '" + decl.name + "'", decl.pos)
                    continue
                if decl.name in RESERVED_NAMES:
                    self.error("cannot use reserved name '" + decl.name + "'", decl.pos)
                    continue
                # Placeholder — fields/methods filled in next loop
                st = StructT(
                    kind="struct",
                    name=decl.name,
                    fields={},
                    methods={},
                    parent=decl.parent,
                    field_order=[],
                )
                self.types[decl.name] = st
            elif isinstance(decl, TInterfaceDecl):
                if decl.name in self.types:
                    self.error("duplicate type name '" + decl.name + "'", decl.pos)
                    continue
                if decl.name in RESERVED_NAMES:
                    self.error("cannot use reserved name '" + decl.name + "'", decl.pos)
                    continue
                it = InterfaceT(kind="interface", name=decl.name, variants=[])
                self.types[decl.name] = it
            elif isinstance(decl, TEnumDecl):
                if decl.name in self.types:
                    self.error("duplicate type name '" + decl.name + "'", decl.pos)
                    continue
                if decl.name in RESERVED_NAMES:
                    self.error("cannot use reserved name '" + decl.name + "'", decl.pos)
                    continue
                seen_variants: set[str] = set()
                for v in decl.variants:
                    if v in seen_variants:
                        self.error(
                            "duplicate variant '" + v + "' in " + decl.name, decl.pos
                        )
                    seen_variants.add(v)
                et = EnumT(kind="enum", name=decl.name, variants=list(decl.variants))
                self.types[decl.name] = et

        # Second pass: resolve struct fields, methods, and interface parents
        for decl in module.decls:
            if isinstance(decl, TStructDecl):
                if decl.name not in self.types:
                    continue
                st2 = self.types[decl.name]
                if not isinstance(st2, StructT):
                    continue
                # Resolve fields
                min_f = 0
                for f in decl.fields:
                    ft = self.resolve_type(f.typ)
                    st2.fields[f.name] = ft
                    st2.field_order.append(f.name)
                    if not f.has_default:
                        min_f += 1
                st2.min_fields = min_f
                # Resolve methods
                for m in decl.methods:
                    mparams: list[Type] = []
                    min_mp = 0
                    for p in m.params:
                        if p.typ is not None:
                            mparams.append(self.resolve_type(p.typ))
                            if not p.has_default:
                                min_mp += 1
                    mret = self.resolve_type(m.ret)
                    st2.methods[m.name] = FnT(
                        kind="fn", params=mparams, ret=mret, min_params=min_mp
                    )
                # Register with parent interface
                if decl.parent is not None:
                    if decl.parent not in self.types:
                        self.error("unknown interface '" + decl.parent + "'", decl.pos)
                    else:
                        parent_type = self.types[decl.parent]
                        if not isinstance(parent_type, InterfaceT):
                            self.error(
                                "'" + decl.parent + "' is not an interface", decl.pos
                            )
                        else:
                            parent_type.variants.append(decl.name)

        # Variant propagation: child interfaces register with parent interfaces
        for decl in module.decls:
            if isinstance(decl, TInterfaceDecl):
                parent_name = decl.annotations.get("_parent_interface", "")
                if parent_name != "":
                    parent_type = self.types.get(parent_name)
                    if parent_type is not None and isinstance(parent_type, InterfaceT):
                        parent_type.variants.append(decl.name)
                        child_type = self.types.get(decl.name)
                        if child_type is not None and isinstance(
                            child_type, InterfaceT
                        ):
                            for v in child_type.variants:
                                parent_type.variants.append(v)

        # Third pass: register top-level functions
        for decl in module.decls:
            if isinstance(decl, TFnDecl):
                if decl.name in self.functions:
                    self.error("duplicate function name '" + decl.name + "'", decl.pos)
                    continue
                if decl.name in self.types:
                    self.error(
                        "'" + decl.name + "' already declared as a type", decl.pos
                    )
                    continue
                if decl.name in RESERVED_NAMES:
                    self.error("cannot use reserved name '" + decl.name + "'", decl.pos)
                    continue
                params2: list[Type] = []
                pnames: list[str] = []
                min_p = 0
                for p in decl.params:
                    pnames.append(p.name)
                    if p.typ is not None:
                        params2.append(self.resolve_type(p.typ))
                        if not p.has_default:
                            min_p += 1
                ret2 = self.resolve_type(decl.ret)
                self.functions[decl.name] = FnT(
                    kind="fn", params=params2, ret=ret2, min_params=min_p
                )
                self.fn_param_names[decl.name] = pnames

    # ── Pass 2: Check bodies ──────────────────────────────────

    def check_bodies(self, module: TModule) -> None:
        for decl in module.decls:
            if isinstance(decl, TFnDecl):
                self.check_fn_decl(decl)
            elif isinstance(decl, TStructDecl):
                self.check_struct_methods(decl)

    def check_fn_decl(self, decl: TFnDecl) -> None:
        ret = self.resolve_type(decl.ret)
        self.current_fn_ret = ret
        saved_uninit = set(self.uninitialized)
        self.uninitialized = set()
        saved_declared = self._declared.copy()
        self._declared = {}
        self.enter_scope()
        for p in decl.params:
            if p.name == "this" and self.current_struct is None:
                self.error("this outside method", p.pos)
                continue
            if p.typ is not None:
                pt = self.resolve_type(p.typ)
                if type_eq(pt, VOID_T):
                    self.error("void is not a value type", p.pos)
                self.declare(p.name, pt, p.pos)
        self.check_stmts(decl.body)
        self.exit_scope()
        if not type_eq(ret, VOID_T) and not _block_is_complete(decl.body):
            self.error("not all paths return a value", decl.pos)
        self.current_fn_ret = None
        self.uninitialized = saved_uninit
        self._declared = saved_declared

    def check_struct_methods(self, decl: TStructDecl) -> None:
        if decl.name not in self.types:
            return
        st = self.types[decl.name]
        if not isinstance(st, StructT):
            return
        old_struct = self.current_struct
        self.current_struct = st
        for method in decl.methods:
            ret = self.resolve_type(method.ret)
            self.current_fn_ret = ret
            saved_uninit = set(self.uninitialized)
            self.uninitialized = set()
            saved_declared = self._declared.copy()
            self._declared = {}
            self.enter_scope()
            # Bind self
            for p in method.params:
                if p.typ is None:
                    self.declare(p.name, st, p.pos)
                else:
                    pt = self.resolve_type(p.typ)
                    self.declare(p.name, pt, p.pos)
            self.check_stmts(method.body)
            self.exit_scope()
            if not type_eq(ret, VOID_T) and not _block_is_complete(method.body):
                self.error("not all paths return a value", method.pos)
            self.uninitialized = saved_uninit
            self._declared = saved_declared
            self.current_fn_ret = None
        self.current_struct = old_struct

    # ── Assignment target validation ─────────────────────────

    def _is_valid_lvalue(self, expr: TExpr) -> bool:
        """Check if expr is a valid assignment target (variable, field, or index)."""
        if isinstance(expr, TVar):
            return True
        if isinstance(expr, TFieldAccess):
            return True
        if isinstance(expr, TIndex):
            return True
        if isinstance(expr, TTupleAccess):
            return True  # syntactically valid, immutability checked separately
        return False

    def _check_assign_target(self, target: TExpr, pos: Pos) -> str | None:
        """Check assignment target for immutability. Returns error msg or None."""
        if isinstance(target, TVar) and target.name == "this":
            return "cannot assign to this"
        if isinstance(target, TVar) and target.name in self.loop_vars:
            return "cannot assign to loop variable"
        if isinstance(target, TTupleAccess):
            return "cannot assign to tuple element"
        if isinstance(target, TIndex):
            obj_type = self.check_expr(target.obj, None)
            if obj_type is not None and type_eq(obj_type, STRING_T):
                return "cannot assign to string index"
            if obj_type is not None and type_eq(obj_type, BYTES_T):
                return "cannot assign to bytes index"
        return None

    # ── Statement checking ────────────────────────────────────

    def check_stmts(self, stmts: list[TStmt]) -> None:
        i = 0
        while i < len(stmts):
            s = stmts[i]
            if i > 0 and isinstance(
                stmts[i - 1], (TReturnStmt, TThrowStmt, TBreakStmt, TContinueStmt)
            ):
                self.error("unreachable code", s.pos)
                return
            self.check_stmt(s)
            # After if-stmt with early exit, narrow nil-checked vars
            if isinstance(s, TIfStmt) and s.else_body is None:
                if _body_always_exits(s.then_body):
                    checks = _collect_nil_checks_guard(s.cond)
                    for var_name, check_kind in checks:
                        if "." in var_name:
                            var_type = self._lookup_field_type(var_name, s.pos)
                        else:
                            var_type = self.lookup(var_name, s.pos)
                        if var_type is not None and contains_nil(var_type):
                            if check_kind == "is_nil":
                                self.narrow(var_name, remove_nil(var_type))
                            elif check_kind == "is_not_nil":
                                self.narrow(var_name, NIL_T)
                    # Guard narrowing for IsType: if !IsType(x, T): return → narrow x to T
                    type_checks = _collect_type_checks_guard(s.cond)
                    for tc_var, tc_type_name, tc_positive in type_checks:
                        if tc_positive:
                            continue
                        if "." in tc_var:
                            current = self._lookup_field_type(tc_var, s.pos)
                        else:
                            current = self.lookup(tc_var, s.pos)
                        if current is None:
                            continue
                        resolved = self._narrow_to_type(current, tc_type_name)
                        if resolved is not None:
                            self.narrow(tc_var, resolved)
            # After Assert(cond), narrow nil-checked and type-checked vars
            if isinstance(s, TExprStmt) and isinstance(s.expr, TCall):
                if isinstance(s.expr.func, TVar) and s.expr.func.name == "Assert":
                    if len(s.expr.args) > 0:
                        assert_cond = s.expr.args[0].value
                        checks = _collect_nil_checks(assert_cond, False, "&&")
                        for var_name, check_kind in checks:
                            if "." in var_name:
                                var_type = self._lookup_field_type(var_name, s.pos)
                            else:
                                var_type = self.lookup(var_name, s.pos)
                            if var_type is not None and contains_nil(var_type):
                                if check_kind == "is_not_nil":
                                    self.narrow(var_name, remove_nil(var_type))
                                elif check_kind == "is_nil":
                                    self.narrow(var_name, NIL_T)
                        type_checks = _collect_type_checks(assert_cond)
                        for tc_var, tc_type_name, tc_positive in type_checks:
                            if not tc_positive:
                                continue
                            if "." in tc_var:
                                current = self._lookup_field_type(tc_var, s.pos)
                            else:
                                current = self.lookup(tc_var, s.pos)
                            if current is None:
                                continue
                            resolved = self._narrow_to_type(current, tc_type_name)
                            if resolved is not None:
                                self.narrow(tc_var, resolved)
            i += 1

    def check_stmt(self, stmt: TStmt) -> None:
        if isinstance(stmt, TLetStmt):
            self.check_let_stmt(stmt)
        elif isinstance(stmt, TAssignStmt):
            self.check_assign_stmt(stmt)
        elif isinstance(stmt, TOpAssignStmt):
            self.check_op_assign_stmt(stmt)
        elif isinstance(stmt, TTupleAssignStmt):
            self.check_tuple_assign_stmt(stmt)
        elif isinstance(stmt, TReturnStmt):
            if self.in_finally:
                self.error("control flow in finally", stmt.pos)
            else:
                self.check_return_stmt(stmt)
        elif isinstance(stmt, TBreakStmt):
            if self.in_finally:
                self.error("control flow in finally", stmt.pos)
            elif not self.in_loop:
                self.error("break outside of loop", stmt.pos)
        elif isinstance(stmt, TContinueStmt):
            if self.in_finally:
                self.error("control flow in finally", stmt.pos)
            elif not self.in_loop:
                self.error("continue outside of loop", stmt.pos)
        elif isinstance(stmt, TThrowStmt):
            if self.in_finally:
                self.error("control flow in finally", stmt.pos)
                return
            throw_type = self.check_expr(stmt.expr, None)
            if throw_type is not None and throw_type.kind != TY_ERROR:
                if not isinstance(throw_type, (StructT, InterfaceT)):
                    if isinstance(throw_type, UnionT):
                        all_throwable = True
                        for m in throw_type.members:
                            if not isinstance(m, (StructT, InterfaceT)):
                                all_throwable = False
                                break
                        if not all_throwable:
                            self.error(
                                "cannot throw " + type_name(throw_type),
                                stmt.pos,
                            )
                    else:
                        self.error(
                            "cannot throw " + type_name(throw_type),
                            stmt.pos,
                        )
        elif isinstance(stmt, TExprStmt):
            # TODO: TStringLit/TNilLit exemption is too broad — should only allow docstrings/noop
            if not isinstance(stmt.expr, (TCall, TStringLit, TNilLit)):
                self.error("expression has no effect", stmt.pos)
            else:
                self.check_expr(stmt.expr, None)
        elif isinstance(stmt, TIfStmt):
            self.check_if_stmt(stmt)
        elif isinstance(stmt, TWhileStmt):
            self.check_while_stmt(stmt)
        elif isinstance(stmt, TForStmt):
            self.check_for_stmt(stmt)
        elif isinstance(stmt, TMatchStmt):
            self.check_match_stmt(stmt)
        elif isinstance(stmt, TTryStmt):
            self.check_try_stmt(stmt)
        else:
            self.error("unhandled statement type", stmt.pos)

    def check_let_stmt(self, stmt: TLetStmt) -> None:
        declared_type = self.resolve_type(stmt.typ)
        if declared_type.kind == TY_VOID:
            self.error("void is not a value type", stmt.pos)
            self.declare(stmt.name, ERROR_T, stmt.pos)
            return
        if stmt.value is not None:
            val_type = self.check_expr(stmt.value, declared_type)
            if val_type is not None and not is_assignable(val_type, declared_type):
                self.error(
                    "cannot assign "
                    + type_name(val_type)
                    + " to "
                    + type_name(declared_type),
                    stmt.pos,
                )
        elif not _has_zero_value(declared_type):
            self.uninitialized.add(stmt.name)
        if stmt.value is not None and type_eq(declared_type, BOOL_T):
            self.bool_facts[stmt.name] = stmt.value
        self.declare(stmt.name, declared_type, stmt.pos)

    def check_assign_stmt(self, stmt: TAssignStmt) -> None:
        if isinstance(stmt.target, TSlice):
            self.error("cannot assign to slice", stmt.pos)
            return
        if not self._is_valid_lvalue(stmt.target):
            self.error("invalid assignment target", stmt.pos)
            return
        immut_err = self._check_assign_target(stmt.target, stmt.pos)
        if immut_err is not None:
            self.error(immut_err, stmt.pos)
            return
        # Direct variable assignment initializes the variable (not a read)
        if isinstance(stmt.target, TVar):
            self.uninitialized.discard(stmt.target.name)
            target_type = self.lookup_declared(stmt.target.name, stmt.pos)
        else:
            target_type = self.check_expr(stmt.target, None)
        if target_type is not None:
            val_type = self.check_expr(stmt.value, target_type)
            if val_type is not None and not is_assignable(val_type, target_type):
                self.error(
                    "cannot assign "
                    + type_name(val_type)
                    + " to "
                    + type_name(target_type),
                    stmt.pos,
                )

    def check_op_assign_stmt(self, stmt: TOpAssignStmt) -> None:
        target_type = self.check_expr(stmt.target, None)
        val_type = self.check_expr(stmt.value, target_type)
        if target_type is not None and val_type is not None:
            # The operator is e.g. "+=" -> underlying op is "+"
            op = stmt.op
            if op.endswith("="):
                op = op[:-1]
            result = self.check_binary_op_types(op, target_type, val_type, stmt.pos)
            if result is not None and not is_assignable(result, target_type):
                self.error(
                    "operator result "
                    + type_name(result)
                    + " not assignable to "
                    + type_name(target_type),
                    stmt.pos,
                )

    def check_tuple_assign_stmt(self, stmt: TTupleAssignStmt) -> None:
        rhs_type = self.check_expr(stmt.value, None)
        if rhs_type is None:
            return
        if rhs_type.kind == TY_ERROR:
            return
        if not isinstance(rhs_type, TupleT):
            self.error(
                "right side of tuple assignment must be a tuple, got "
                + type_name(rhs_type),
                stmt.pos,
            )
            return
        if len(rhs_type.elements) != len(stmt.targets):
            self.error(
                "tuple assignment arity mismatch: "
                + str(len(stmt.targets))
                + " targets, "
                + str(len(rhs_type.elements))
                + " values",
                stmt.pos,
            )
            return
        i = 0
        while i < len(stmt.targets):
            tgt = stmt.targets[i]
            if isinstance(tgt, TVar) and tgt.name == "_":
                i += 1
                continue
            if isinstance(tgt, TVar):
                self.uninitialized.discard(tgt.name)
                target_type = self.lookup_declared(tgt.name, stmt.pos)
            else:
                target_type = self.check_expr(tgt, None)
            if target_type is not None and not is_assignable(
                rhs_type.elements[i], target_type
            ):
                self.error(
                    "cannot assign "
                    + type_name(rhs_type.elements[i])
                    + " to "
                    + type_name(target_type),
                    stmt.pos,
                )
            i += 1

    def check_return_stmt(self, stmt: TReturnStmt) -> None:
        if self.current_fn_ret is None:
            self.error("return outside of function", stmt.pos)
            return
        if stmt.value is None:
            if not type_eq(self.current_fn_ret, VOID_T):
                self.error("missing return value in non-void function", stmt.pos)
        else:
            if type_eq(self.current_fn_ret, VOID_T):
                self.error("cannot return value from void function", stmt.pos)
                return
            val_type = self.check_expr(stmt.value, self.current_fn_ret)
            if val_type is not None and not is_assignable(
                val_type, self.current_fn_ret
            ):
                self.error(
                    "cannot return "
                    + type_name(val_type)
                    + " from function returning "
                    + type_name(self.current_fn_ret),
                    stmt.pos,
                )

    def check_if_stmt(self, stmt: TIfStmt) -> None:
        cond_type = self.check_expr(stmt.cond, BOOL_T)
        if (
            cond_type is not None
            and cond_type.kind != TY_ERROR
            and not type_eq(cond_type, BOOL_T)
            and not _is_truthy_type(cond_type)
        ):
            self.error(
                "if condition must be bool, got " + type_name(cond_type), stmt.pos
            )
        # Nil narrowing in then/else bodies via == nil / != nil checks
        # (IsNil deliberately excluded — it doesn't narrow in then-body)
        narrowings: list[tuple[str, Type, Type]] = []
        narrow_cond = self._resolve_bool_var(stmt.cond)
        var_checks = _collect_nil_checks(narrow_cond, True, "&&")
        all_checks = _collect_nil_checks(narrow_cond, False, "&&")
        checks: list[tuple[str, str]] = []
        fc_i = 0
        while fc_i < len(var_checks):
            checks.append(var_checks[fc_i])
            fc_i += 1
        fc_i = 0
        while fc_i < len(all_checks):
            n, k = all_checks[fc_i]
            if (n, k) not in var_checks:
                checks.append((n, k))
            fc_i += 1
        for var_name, check_kind in checks:
            if "." in var_name:
                var_type = self._lookup_field_type(var_name, stmt.pos)
            else:
                var_type = self.lookup(var_name, stmt.pos)
            if var_type is not None and contains_nil(var_type):
                if check_kind == "is_not_nil":
                    narrowings.append((var_name, remove_nil(var_type), NIL_T))
                elif check_kind == "is_nil":
                    narrowings.append((var_name, NIL_T, remove_nil(var_type)))
        # IsType narrowing: isinstance(x, T) → narrow x to T in then, keep original in else
        type_checks = _collect_type_checks(narrow_cond)
        for tc_var, tc_type_name, tc_positive in type_checks:
            if "." in tc_var:
                current = self._lookup_field_type(tc_var, stmt.pos)
            else:
                current = self.lookup(tc_var, stmt.pos)
            if current is None:
                continue
            resolved = self._narrow_to_type(current, tc_type_name)
            if resolved is None:
                continue
            if tc_positive:
                narrowings.append((tc_var, resolved, current))
            else:
                narrowings.append((tc_var, current, resolved))
        # Check then-body with narrowing
        saved_uninit = set(self.uninitialized)
        self.enter_scope()
        for name, then_type, _else_type in narrowings:
            self.scopes[-1][name] = then_type
        self.check_stmts(stmt.then_body)
        self.exit_scope()
        then_uninit = set(self.uninitialized)
        # Check else-body with reverse narrowing
        self.uninitialized = set(saved_uninit)
        if stmt.else_body is not None:
            self.enter_scope()
            for name, _then_type, else_type in narrowings:
                self.scopes[-1][name] = else_type
            self.check_stmts(stmt.else_body)
            self.exit_scope()
        else_uninit = set(self.uninitialized)
        # Merge: initialized only if initialized in BOTH branches
        then_exits = _block_always_exits(stmt.then_body)
        else_exits = stmt.else_body is not None and _block_always_exits(stmt.else_body)
        if then_exits and else_exits:
            self.uninitialized = saved_uninit
        elif then_exits:
            self.uninitialized = else_uninit
        elif else_exits:
            self.uninitialized = then_uninit
        else:
            self.uninitialized = then_uninit | else_uninit

    def check_while_stmt(self, stmt: TWhileStmt) -> None:
        cond_type = self.check_expr(stmt.cond, BOOL_T)
        if (
            cond_type is not None
            and cond_type.kind != TY_ERROR
            and not type_eq(cond_type, BOOL_T)
            and not _is_truthy_type(cond_type)
        ):
            self.error(
                "while condition must be bool, got " + type_name(cond_type), stmt.pos
            )
        old_in_loop = self.in_loop
        self.in_loop = True
        saved_uninit = set(self.uninitialized)
        self.enter_scope()
        narrow_cond = self._resolve_bool_var(stmt.cond)
        nil_checks = _collect_nil_checks(narrow_cond, False, "&&")
        for var_name, check_kind in nil_checks:
            if "." in var_name:
                var_type = self._lookup_field_type(var_name, stmt.pos)
            else:
                var_type = self.lookup(var_name, stmt.pos)
            if var_type is not None and contains_nil(var_type):
                if check_kind == "is_not_nil":
                    self.narrow(var_name, remove_nil(var_type))
                elif check_kind == "is_nil":
                    self.narrow(var_name, NIL_T)
        self.check_stmts(stmt.body)
        self.exit_scope()
        self.uninitialized = saved_uninit
        self.in_loop = old_in_loop

    def check_for_stmt(self, stmt: TForStmt) -> None:
        old_in_loop = self.in_loop
        self.in_loop = True
        old_loop_vars = self.loop_vars
        self.loop_vars = set(b for b in stmt.binding if b != "_")
        saved_uninit = set(self.uninitialized)
        self.enter_scope()
        if isinstance(stmt.iterable, TRange):
            # range — all args must be int, loop var is int
            for arg in stmt.iterable.args:
                arg_type = self.check_expr(arg, INT_T)
                if arg_type is not None and not type_eq(arg_type, INT_T):
                    self.error(
                        "range argument must be int, got " + type_name(arg_type),
                        stmt.iterable.pos,
                    )
            if len(stmt.binding) == 1:
                self.declare(stmt.binding[0], INT_T, stmt.pos)
            elif len(stmt.binding) == 2:
                self.error("range does not support the two-variable form", stmt.pos)
                self.declare(stmt.binding[0], INT_T, stmt.pos)
                self.declare(stmt.binding[1], INT_T, stmt.pos)
        else:
            iter_type = self.check_expr(stmt.iterable, None)
            if iter_type is not None:
                if iter_type.kind == TY_ERROR:
                    for b in stmt.binding:
                        self.declare(b, ERROR_T, stmt.pos)
                else:
                    self.bind_for_vars(stmt.binding, iter_type, stmt.pos)
                    if isinstance(iter_type, MapT):
                        stmt.annotations["iter_kind"] = "map"
                    elif isinstance(iter_type, ListT) and len(stmt.binding) >= 2:
                        if isinstance(iter_type.element, TupleT) and len(
                            iter_type.element.elements
                        ) == len(stmt.binding):
                            stmt.annotations["iter_kind"] = "tuple_unpack"
                        elif len(stmt.binding) == 2:
                            stmt.annotations["iter_kind"] = "enumerate"
        self.check_stmts(stmt.body)
        self.exit_scope()
        self.uninitialized = saved_uninit
        self.in_loop = old_in_loop
        self.loop_vars = old_loop_vars

    def bind_for_vars(self, binding: list[str], iter_type: Type, pos: Pos) -> None:
        if isinstance(iter_type, ListT):
            if len(binding) == 1:
                self.declare(binding[0], iter_type.element, pos)
            elif (
                len(binding) == 2
                and isinstance(iter_type.element, TupleT)
                and len(iter_type.element.elements) == 2
            ):
                self.declare(binding[0], iter_type.element.elements[0], pos)
                self.declare(binding[1], iter_type.element.elements[1], pos)
            elif len(binding) == 2:
                self.declare(binding[0], INT_T, pos)
                self.declare(binding[1], iter_type.element, pos)
            elif isinstance(iter_type.element, TupleT) and len(
                iter_type.element.elements
            ) == len(binding):
                i = 0
                while i < len(binding):
                    self.declare(binding[i], iter_type.element.elements[i], pos)
                    i += 1
        elif type_eq(iter_type, STRING_T):
            if len(binding) == 1:
                self.declare(binding[0], RUNE_T, pos)
            elif len(binding) == 2:
                self.declare(binding[0], INT_T, pos)
                self.declare(binding[1], RUNE_T, pos)
        elif type_eq(iter_type, BYTES_T):
            if len(binding) == 1:
                self.declare(binding[0], BYTE_T, pos)
            elif len(binding) == 2:
                self.declare(binding[0], INT_T, pos)
                self.declare(binding[1], BYTE_T, pos)
        elif isinstance(iter_type, MapT):
            if len(binding) == 1:
                self.declare(binding[0], iter_type.key, pos)
            elif len(binding) == 2:
                self.declare(binding[0], iter_type.key, pos)
                self.declare(binding[1], iter_type.value, pos)
        elif isinstance(iter_type, SetT):
            if len(binding) == 1:
                self.declare(binding[0], iter_type.element, pos)
            elif len(binding) == 2:
                self.error("sets do not support the two-variable for form", pos)
                self.declare(binding[0], iter_type.element, pos)
                self.declare(binding[1], iter_type.element, pos)
        else:
            self.error("cannot iterate over " + type_name(iter_type), pos)
            for b in binding:
                self.declare(b, ERROR_T, pos)

    def check_match_stmt(self, stmt: TMatchStmt) -> None:
        scrutinee_type = self.check_expr(stmt.expr, None)
        if scrutinee_type is None:
            return
        if not self._is_matchable(scrutinee_type):
            self.error("cannot match on " + type_name(scrutinee_type), stmt.pos)
            return
        saved_uninit = set(self.uninitialized)
        covered: list[str] = []
        case_uninits: list[set[str]] = []
        for case in stmt.cases:
            self.uninitialized = set(saved_uninit)
            self.check_match_case(case, scrutinee_type, covered, stmt.expr)
            case_uninits.append(set(self.uninitialized))
        has_default = stmt.default is not None
        if has_default:
            dflt = stmt.default
            assert dflt is not None
            self.uninitialized = set(saved_uninit)
            self.enter_scope()
            if dflt.name is not None:
                residual = self._compute_default_type(scrutinee_type, covered)
                self.declare(dflt.name, residual, dflt.pos)
            self.check_stmts(dflt.body)
            self.exit_scope()
            case_uninits.append(set(self.uninitialized))
        exhaustive = has_default
        if not has_default:
            err_count = len(self.errors)
            self.check_exhaustiveness(scrutinee_type, covered, stmt.pos)
            exhaustive = len(self.errors) == err_count
        # Merge: if exhaustive, var is initialized only if ALL branches initialized it
        if exhaustive and len(case_uninits) > 0:
            merged: set[str] = set()
            for v in case_uninits[0]:
                merged.add(v)
            cui = 1
            while cui < len(case_uninits):
                for v in case_uninits[cui]:
                    merged.add(v)
                cui += 1
            result_uninit: set[str] = set()
            for v in merged:
                if v in saved_uninit:
                    result_uninit.add(v)
            self.uninitialized = result_uninit
        else:
            self.uninitialized = saved_uninit

    def _is_matchable(self, t: Type) -> bool:
        """Check if a type is valid as a match scrutinee."""
        if isinstance(t, (InterfaceT, EnumT)):
            return True
        if isinstance(t, UnionT):
            return True
        if isinstance(t, StructT):
            return True
        if t.kind == TY_ERROR:
            return True
        # Primitive types and collection types — degenerate but valid
        return True

    def _allowed_in_match(self, case_type: Type, scrutinee: Type) -> bool:
        """Check if case_type is a valid case for the given scrutinee."""
        if isinstance(scrutinee, InterfaceT):
            if isinstance(case_type, (StructT, InterfaceT)):
                return case_type.name in scrutinee.variants
            return False
        if isinstance(scrutinee, EnumT):
            return False  # enums use TPatternEnum, not TPatternType
        if isinstance(scrutinee, UnionT):
            for m in scrutinee.members:
                if type_eq(case_type, m):
                    return True
                if isinstance(m, InterfaceT):
                    if isinstance(case_type, StructT) and case_type.name in m.variants:
                        return True
                    if (
                        isinstance(case_type, InterfaceT)
                        and case_type.name in m.variants
                    ):
                        return True
            return False
        if isinstance(scrutinee, MapT):
            return True
        if scrutinee.kind == TY_ERROR:
            return True
        # Struct/primitive/collection scrutinee: allow any struct/type case
        if isinstance(scrutinee, StructT):
            return isinstance(case_type, StructT)
        return type_eq(case_type, scrutinee)

    def check_match_case(
        self,
        case: TMatchCase,
        scrutinee: Type,
        covered: list[str],
        scrutinee_expr: TExpr | None = None,
    ) -> None:
        pat = case.pattern
        if isinstance(pat, TPatternNil):
            if not contains_nil(scrutinee):
                self.error(
                    "nil is not a variant of " + type_name(scrutinee),
                    pat.pos,
                )
            key = "nil"
            if key in covered:
                self.error("duplicate case: nil", pat.pos)
            covered.append(key)
            self.enter_scope()
            self.check_stmts(case.body)
            self.exit_scope()
        elif isinstance(pat, TPatternEnum):
            # Validate enum — check that this enum is valid for the scrutinee
            if pat.enum_name in self.types:
                enum_type = self.types[pat.enum_name]
                if isinstance(enum_type, EnumT):
                    # Check that this enum is the right one for the scrutinee
                    scrutinee_enum = self._get_scrutinee_enum(scrutinee)
                    if scrutinee_enum is not None and not type_eq(
                        enum_type, scrutinee_enum
                    ):
                        self.error(
                            "'"
                            + pat.enum_name
                            + "."
                            + pat.variant
                            + "' is not a variant of "
                            + scrutinee_enum.name,
                            pat.pos,
                        )
                    elif pat.variant not in enum_type.variants:
                        self.error(
                            "'"
                            + pat.variant
                            + "' is not a variant of "
                            + pat.enum_name,
                            pat.pos,
                        )
                    else:
                        key = pat.enum_name + "." + pat.variant
                        if key in covered:
                            self.error("duplicate case", pat.pos)
                        covered.append(key)
                else:
                    self.error("'" + pat.enum_name + "' is not an enum", pat.pos)
            else:
                self.error("unknown type '" + pat.enum_name + "'", pat.pos)
            self.enter_scope()
            self.check_stmts(case.body)
            self.exit_scope()
        elif isinstance(pat, TPatternType):
            case_type = self.resolve_type(pat.type_name)
            if (
                case_type.kind != TY_ERROR
                and scrutinee.kind != TY_ERROR
                and not self._allowed_in_match(case_type, scrutinee)
            ):
                self.error(
                    type_name(case_type)
                    + " is not a variant of "
                    + type_name(scrutinee),
                    pat.pos,
                )
            key2 = _type_key(case_type)
            if key2 in covered:
                self.error("duplicate case", pat.pos)
            covered.append(key2)
            # For interface cases in a union, also mark all struct variants covered
            if isinstance(case_type, InterfaceT):
                for v in case_type.variants:
                    vkey = _type_key(self.types[v])
                    if vkey not in covered:
                        covered.append(vkey)
            self.enter_scope()
            self.declare(pat.name, case_type, pat.pos)
            if scrutinee_expr is not None and isinstance(scrutinee_expr, TVar):
                if scrutinee_expr.name != pat.name:
                    self.narrow(scrutinee_expr.name, case_type)
            self.check_stmts(case.body)
            self.exit_scope()

    def _get_scrutinee_enum(self, scrutinee: Type) -> EnumT | None:
        """Get the enum type from the scrutinee (direct or optional)."""
        if isinstance(scrutinee, EnumT):
            return scrutinee
        if isinstance(scrutinee, UnionT):
            for m in scrutinee.members:
                if isinstance(m, EnumT):
                    return m
        return None

    def _compute_default_type(self, scrutinee: Type, covered: list[str]) -> Type:
        """Compute residual type for a default binding (scrutinee minus covered)."""
        if isinstance(scrutinee, InterfaceT):
            remaining: list[Type] = []
            for v in scrutinee.variants:
                vt = self.types[v]
                if isinstance(vt, InterfaceT):
                    continue
                if _type_key(vt) not in covered:
                    remaining.append(vt)
            if len(remaining) == 0:
                return scrutinee
            if len(remaining) == 1:
                return remaining[0]
            return normalize_union(remaining)
        if isinstance(scrutinee, UnionT):
            remaining2: list[Type] = []
            for m in scrutinee.members:
                if type_eq(m, NIL_T):
                    if "nil" not in covered:
                        remaining2.append(m)
                elif isinstance(m, InterfaceT):
                    for v in m.variants:
                        vt2 = self.types[v]
                        if isinstance(vt2, InterfaceT):
                            continue
                        if _type_key(vt2) not in covered:
                            remaining2.append(vt2)
                else:
                    if _type_key(m) not in covered:
                        remaining2.append(m)
            if len(remaining2) == 0:
                return scrutinee
            if len(remaining2) == 1:
                return remaining2[0]
            return normalize_union(remaining2)
        if isinstance(scrutinee, EnumT):
            return scrutinee
        return ERROR_T

    def check_exhaustiveness(
        self, scrutinee: Type, covered: list[str], pos: Pos
    ) -> None:
        required: list[str] = []
        if isinstance(scrutinee, InterfaceT):
            for v in scrutinee.variants:
                vt = self.types[v]
                if isinstance(vt, InterfaceT):
                    continue
                required.append(_type_key(vt))
        elif isinstance(scrutinee, EnumT):
            for v in scrutinee.variants:
                required.append(scrutinee.name + "." + v)
        elif isinstance(scrutinee, UnionT):
            for m in scrutinee.members:
                if type_eq(m, NIL_T):
                    required.append("nil")
                elif isinstance(m, InterfaceT):
                    for v in m.variants:
                        vt = self.types[v]
                        if isinstance(vt, InterfaceT):
                            continue
                        required.append(_type_key(vt))
                elif isinstance(m, EnumT):
                    for v in m.variants:
                        required.append(m.name + "." + v)
                else:
                    required.append(_type_key(m))
        elif type_eq(scrutinee, ERROR_T):
            self.error("non-exhaustive match: default case required", pos)
            return
        else:
            return
        missing: list[str] = []
        for r in required:
            if r not in covered:
                missing.append(r)
        if len(missing) > 0:
            self.error("non-exhaustive match: missing cases", pos)

    def check_try_stmt(self, stmt: TTryStmt) -> None:
        saved_uninit = set(self.uninitialized)
        self.enter_scope()
        self.check_stmts(stmt.body)
        self.exit_scope()
        seen_catch_all = False
        seen_catch_types: list[Type] = []
        for catch in stmt.catches:
            self.enter_scope()
            if len(catch.types) == 0:
                catch_type = ERROR_T
            elif len(catch.types) == 1:
                catch_type = self.resolve_type(catch.types[0])
            else:
                members: list[Type] = []
                for ct in catch.types:
                    members.append(self.resolve_type(ct))
                catch_type = normalize_union(members)
            if seen_catch_all:
                self.error("unreachable catch after catch-all", catch.pos)
            if type_eq(catch_type, ERROR_T):
                seen_catch_all = True
            elif not isinstance(catch_type, (StructT, InterfaceT, UnionT)):
                self.error(
                    "catch type must be a struct, got " + type_name(catch_type),
                    catch.pos,
                )
            else:
                for prev in seen_catch_types:
                    if type_eq(catch_type, prev):
                        self.error(
                            "duplicate catch for " + type_name(catch_type), catch.pos
                        )
                        break
                seen_catch_types.append(catch_type)
            self.declare(catch.name, catch_type, catch.pos)
            self.check_stmts(catch.body)
            self.exit_scope()
        if stmt.finally_body is not None:
            old_in_finally = self.in_finally
            self.in_finally = True
            self.enter_scope()
            self.check_stmts(stmt.finally_body)
            self.exit_scope()
            self.in_finally = old_in_finally
        self.uninitialized = saved_uninit

    # ── Expression checking ───────────────────────────────────

    def check_expr(self, expr: TExpr, expected: Type | None) -> Type | None:
        """Type-check an expression and return its type. Returns None on error."""
        result = self._check_expr_inner(expr, expected)
        if result is not None:
            self.expr_types[(expr.pos.line, expr.pos.col)] = result
            expr.annotations["type"] = type_name(result)
        return result

    def _check_expr_inner(self, expr: TExpr, expected: Type | None) -> Type | None:
        if isinstance(expr, TIntLit):
            if len(expr.raw) > 20 or (
                len(expr.raw) == 20 and expr.raw > "18446744073709551615"
            ):
                self.error("integer literal too large", expr.pos)
            return INT_T
        if isinstance(expr, TFloatLit):
            return FLOAT_T
        if isinstance(expr, TBoolLit):
            return BOOL_T
        if isinstance(expr, TByteLit):
            if expected is not None and type_eq(expected, INT_T):
                return INT_T
            return BYTE_T
        if isinstance(expr, TStringLit):
            return STRING_T
        if isinstance(expr, TRuneLit):
            return RUNE_T
        if isinstance(expr, TBytesLit):
            return BYTES_T
        if isinstance(expr, TNilLit):
            return NIL_T
        if isinstance(expr, TVar):
            return self.check_var(expr)
        if isinstance(expr, TBinaryOp):
            return self.check_binary_op(expr)
        if isinstance(expr, TUnaryOp):
            return self.check_unary_op(expr)
        if isinstance(expr, TTernary):
            return self.check_ternary(expr, expected)
        if isinstance(expr, TFieldAccess):
            return self.check_field_access(expr)
        if isinstance(expr, TTupleAccess):
            return self.check_tuple_access(expr)
        if isinstance(expr, TIndex):
            return self.check_index(expr)
        if isinstance(expr, TSlice):
            return self.check_slice(expr)
        if isinstance(expr, TCall):
            return self.check_call(expr, expected)
        if isinstance(expr, TListLit):
            return self.check_list_lit(expr, expected)
        if isinstance(expr, TMapLit):
            return self.check_map_lit(expr, expected)
        if isinstance(expr, TSetLit):
            return self.check_set_lit(expr, expected)
        if isinstance(expr, TTupleLit):
            return self.check_tuple_lit(expr, expected)
        if isinstance(expr, TFnLit):
            return self.check_fn_lit(expr, expected)
        self.error("unhandled expression type", expr.pos)
        return None

    def check_var(self, expr: TVar) -> Type | None:
        if expr.name in self.uninitialized:
            self.error("variable used before assignment", expr.pos)
        return self.lookup(expr.name, expr.pos)

    def check_binary_op(self, expr: TBinaryOp) -> Type | None:
        left = self.check_expr(expr.left, None)
        if expr.op == "&&":
            checks = _collect_nil_checks(expr.left, False, "&&")
            tc = _collect_type_checks(expr.left)
            if len(checks) > 0 or len(tc) > 0:
                self.enter_scope()
                for var_name, check_kind in checks:
                    if "." in var_name:
                        var_type = self._lookup_field_type(var_name, expr.pos)
                    else:
                        var_type = self.lookup(var_name, expr.pos)
                    if var_type is not None and contains_nil(var_type):
                        if check_kind == "is_not_nil":
                            self.scopes[-1][var_name] = remove_nil(var_type)
                        elif check_kind == "is_nil":
                            self.scopes[-1][var_name] = NIL_T
                for tc_var, tc_type_name, tc_positive in tc:
                    if not tc_positive:
                        continue
                    if "." in tc_var:
                        current = self._lookup_field_type(tc_var, expr.pos)
                    else:
                        current = self.lookup(tc_var, expr.pos)
                    if current is not None:
                        resolved = self._narrow_to_type(current, tc_type_name)
                        if resolved is not None:
                            self.scopes[-1][tc_var] = resolved
                right = self.check_expr(expr.right, None)
                self.exit_scope()
            else:
                right = self.check_expr(expr.right, None)
        elif expr.op == "||":
            checks = _collect_nil_checks(expr.left, False, "||")
            tc = _collect_type_checks(expr.left)
            if len(checks) > 0 or len(tc) > 0:
                self.enter_scope()
                for var_name, check_kind in checks:
                    if "." in var_name:
                        var_type = self._lookup_field_type(var_name, expr.pos)
                    else:
                        var_type = self.lookup(var_name, expr.pos)
                    if var_type is not None and contains_nil(var_type):
                        if check_kind == "is_nil":
                            self.scopes[-1][var_name] = remove_nil(var_type)
                        elif check_kind == "is_not_nil":
                            self.scopes[-1][var_name] = NIL_T
                # ||: if left is !IsType(x,T) (negative), right sees x narrowed to T
                for tc_var, tc_type_name, tc_positive in tc:
                    if tc_positive:
                        continue
                    if "." in tc_var:
                        current = self._lookup_field_type(tc_var, expr.pos)
                    else:
                        current = self.lookup(tc_var, expr.pos)
                    if current is not None:
                        resolved = self._narrow_to_type(current, tc_type_name)
                        if resolved is not None:
                            self.scopes[-1][tc_var] = resolved
                right = self.check_expr(expr.right, None)
                self.exit_scope()
            else:
                right = self.check_expr(expr.right, None)
        else:
            right_expected = left if expr.op in ("==", "!=") else None
            right = self.check_expr(expr.right, right_expected)
        if left is None or right is None:
            return None
        return self.check_binary_op_types(expr.op, left, right, expr.pos)

    def check_binary_op_types(
        self, op: str, left: Type, right: Type, pos: Pos
    ) -> Type | None:
        if left.kind == TY_ERROR or right.kind == TY_ERROR:
            return ERROR_T
        # Logical: both bool
        if op in ("&&", "||"):
            if not type_eq(left, BOOL_T):
                self.error(
                    "left operand of " + op + " must be bool, got " + type_name(left),
                    pos,
                )
                return None
            if not type_eq(right, BOOL_T):
                self.error(
                    "right operand of " + op + " must be bool, got " + type_name(right),
                    pos,
                )
                return None
            return BOOL_T
        # Equality: same type (or compatible — tuples, lists, error)
        if op in ("==", "!="):
            if _contains_fn_type(left) or _contains_fn_type(right):
                self.error("equality not defined for fn", pos)
                return None
            if not type_eq(left, right) and not (
                is_assignable(left, right) or is_assignable(right, left)
            ):
                # Allow comparing tuples of different sizes
                if isinstance(left, TupleT) and isinstance(right, TupleT):
                    return BOOL_T
                # Allow comparing any two lists
                if isinstance(left, ListT) and isinstance(right, ListT):
                    return BOOL_T
                # Allow comparing any two maps
                if isinstance(left, MapT) and isinstance(right, MapT):
                    return BOOL_T
                # Allow comparing any two sets
                if isinstance(left, SetT) and isinstance(right, SetT):
                    return BOOL_T
                # Allow comparing tuple and list (single-element tuple lowered to list)
                if isinstance(left, (TupleT, ListT)) and isinstance(
                    right, (TupleT, ListT)
                ):
                    return BOOL_T
                # Allow comparing with error type
                if left.kind == TY_ERROR or right.kind == TY_ERROR:
                    return BOOL_T
                # Allow byte/int interchangeability
                if (left.kind == TY_BYTE and right.kind == TY_INT) or (
                    left.kind == TY_INT and right.kind == TY_BYTE
                ):
                    return BOOL_T
                # Allow comparing optional/union-containing-nil with nil
                if left.kind == TY_NIL and contains_nil(right):
                    return BOOL_T
                if right.kind == TY_NIL and contains_nil(left):
                    return BOOL_T
                self.error(
                    "cannot compare " + type_name(left) + " and " + type_name(right),
                    pos,
                )
                return None
            return BOOL_T
        # Ordering: int, float, byte, rune, string, list, tuple — same type
        if op in ("<", "<=", ">", ">="):
            if not type_eq(left, right):
                # Allow comparing tuples of different sizes, or tuple with list
                if isinstance(left, (TupleT, ListT)) and isinstance(
                    right, (TupleT, ListT)
                ):
                    return BOOL_T
                # Allow comparing with error type
                if left.kind == TY_ERROR or right.kind == TY_ERROR:
                    return BOOL_T
                # Allow byte/int interchangeability
                if (left.kind == TY_BYTE and right.kind == TY_INT) or (
                    left.kind == TY_INT and right.kind == TY_BYTE
                ):
                    return BOOL_T
                self.error(
                    "cannot compare " + type_name(left) + " and " + type_name(right),
                    pos,
                )
                return None
            if left.kind not in (
                TY_INT,
                TY_FLOAT,
                TY_BYTE,
                TY_RUNE,
                TY_STRING,
            ):
                msg = (
                    "not defined for union"
                    if isinstance(left, UnionT)
                    else "ordering not defined for " + type_name(left)
                )
                self.error(msg, pos)
                return None
            return BOOL_T
        # Arithmetic: +, -, *, /, % — int, float, byte; same type
        if op in ("+", "-", "*", "/", "%"):
            if not type_eq(left, right):
                self.error(
                    "operands of "
                    + op
                    + " must be same type, got "
                    + type_name(left)
                    + " and "
                    + type_name(right),
                    pos,
                )
                return None
            if left.kind not in (TY_INT, TY_FLOAT, TY_BYTE):
                self.error(op + " not defined for " + type_name(left), pos)
                return None
            return left
        # Bitwise: &, |, ^ — int, byte; same type
        if op in ("&", "|", "^"):
            if not type_eq(left, right):
                self.error(
                    "operands of "
                    + op
                    + " must be same type, got "
                    + type_name(left)
                    + " and "
                    + type_name(right),
                    pos,
                )
                return None
            if left.kind not in (TY_INT, TY_BYTE):
                self.error(op + " not defined for " + type_name(left), pos)
                return None
            return left
        # Shifts: <<, >>, >>> — left is int/byte, right is int
        if op == ">>>" and not self.strict_math:
            self.error(">>> requires strict_math mode", pos)
            return None
        if op in ("<<", ">>", ">>>"):
            if left.kind not in (TY_INT, TY_BYTE):
                self.error(
                    "left operand of "
                    + op
                    + " must be int or byte, got "
                    + type_name(left),
                    pos,
                )
                return None
            if not type_eq(right, INT_T):
                self.error(
                    "right operand of " + op + " must be int, got " + type_name(right),
                    pos,
                )
                return None
            return left
        self.error("unknown binary operator: " + op, pos)
        return None

    def check_unary_op(self, expr: TUnaryOp) -> Type | None:
        operand = self.check_expr(expr.operand, None)
        if operand is None:
            return None
        if operand.kind == TY_ERROR:
            return ERROR_T
        if expr.op == "-":
            if operand.kind not in (TY_INT, TY_FLOAT, TY_BYTE):
                self.error("negation not defined for " + type_name(operand), expr.pos)
                return None
            return operand
        if expr.op == "!":
            if not type_eq(operand, BOOL_T) and not _is_truthy_type(operand):
                self.error(
                    "logical not requires bool, got " + type_name(operand), expr.pos
                )
                return None
            return BOOL_T
        if expr.op == "~":
            if operand.kind not in (TY_INT, TY_BYTE):
                self.error(
                    "bitwise complement not defined for " + type_name(operand), expr.pos
                )
                return None
            return operand
        self.error("unknown unary operator: " + expr.op, expr.pos)
        return None

    def check_ternary(self, expr: TTernary, expected: Type | None) -> Type | None:
        cond = self.check_expr(expr.cond, BOOL_T)
        if (
            cond is not None
            and cond.kind != TY_ERROR
            and not type_eq(cond, BOOL_T)
            and not _is_truthy_type(cond)
        ):
            self.error(
                "ternary condition must be bool, got " + type_name(cond), expr.pos
            )
        # Nil narrowing (same pattern as check_if_stmt)
        narrowings: list[tuple[str, Type, Type]] = []
        var_checks = _collect_nil_checks(expr.cond, True, "&&")
        all_checks = _collect_nil_checks(expr.cond, False, "&&")
        checks: list[tuple[str, str]] = []
        fc_i = 0
        while fc_i < len(var_checks):
            checks.append(var_checks[fc_i])
            fc_i += 1
        fc_i = 0
        while fc_i < len(all_checks):
            n, k = all_checks[fc_i]
            if (n, k) not in var_checks:
                checks.append((n, k))
            fc_i += 1
        for var_name, check_kind in checks:
            if "." in var_name:
                var_type = self._lookup_field_type(var_name, expr.pos)
            else:
                var_type = self.lookup(var_name, expr.pos)
            if var_type is not None and contains_nil(var_type):
                if check_kind == "is_not_nil":
                    narrowings.append((var_name, remove_nil(var_type), NIL_T))
                elif check_kind == "is_nil":
                    narrowings.append((var_name, NIL_T, remove_nil(var_type)))
        # IsType narrowing
        type_checks = _collect_type_checks(expr.cond)
        for tc_var, tc_type_name, tc_positive in type_checks:
            if "." in tc_var:
                current = self._lookup_field_type(tc_var, expr.pos)
            else:
                current = self.lookup(tc_var, expr.pos)
            if current is None:
                continue
            resolved = self._narrow_to_type(current, tc_type_name)
            if resolved is None:
                continue
            if tc_positive:
                narrowings.append((tc_var, resolved, current))
            else:
                narrowings.append((tc_var, current, resolved))
        self.enter_scope()
        for name, then_t, _else_t in narrowings:
            self.scopes[-1][name] = then_t
        then_type = self.check_expr(expr.then_expr, expected)
        self.exit_scope()
        self.enter_scope()
        for name, _then_t, else_t in narrowings:
            self.scopes[-1][name] = else_t
        else_type = self.check_expr(expr.else_expr, expected)
        self.exit_scope()
        if then_type is None or else_type is None:
            return then_type if then_type is not None else else_type
        if not type_eq(then_type, else_type):
            # Allow if one is assignable to the other
            if is_assignable(then_type, else_type):
                return else_type
            if is_assignable(else_type, then_type):
                return then_type
            # Try widening to common interface
            widened = self._widen_to_common_interface(then_type, else_type)
            if widened is not None:
                return widened
            # T vs nil → optional
            if then_type.kind == TY_NIL:
                return UnionT(kind="union", members=[else_type, NIL_T])
            if else_type.kind == TY_NIL:
                return UnionT(kind="union", members=[then_type, NIL_T])
            # Suppress when either side is error
            if then_type.kind == TY_ERROR or else_type.kind == TY_ERROR:
                return ERROR_T
            # Different types → union
            return normalize_union([then_type, else_type])
        return then_type

    def check_field_access(self, expr: TFieldAccess) -> Type | None:
        # Special case: Enum.Variant — TVar.field
        if isinstance(expr.obj, TVar):
            resolved = self.lookup(expr.obj.name, expr.obj.pos)
            if resolved is not None and isinstance(resolved, EnumT):
                if expr.field not in resolved.variants:
                    self.error(
                        "'" + expr.field + "' is not a variant of " + resolved.name,
                        expr.pos,
                    )
                    return None
                return resolved
        obj_type = self.check_expr(expr.obj, None)
        if obj_type is None:
            return None
        path = _field_access_path(expr)
        if path is not None:
            narrowed = self._lookup_narrowed_path(path)
            if narrowed is not None:
                return narrowed
        if isinstance(obj_type, StructT):
            if expr.field in obj_type.fields:
                return obj_type.fields[expr.field]
            if expr.field in obj_type.methods:
                self.error("cannot capture 'this'", expr.pos)
                return obj_type.methods[expr.field]
            self.error(
                "'" + obj_type.name + "' has no field or method '" + expr.field + "'",
                expr.pos,
            )
            return None
        if isinstance(obj_type, UnionT):
            field_type = self._union_field_type(obj_type, expr.field)
            if field_type is not None:
                return field_type
        if isinstance(obj_type, InterfaceT):
            iface_ft = self._interface_field_type(obj_type, expr.field)
            if iface_ft is not None:
                return iface_ft
            self.error(
                "cannot access field on interface '"
                + obj_type.name
                + "'; use match to narrow",
                expr.pos,
            )
            return None
        if obj_type.kind == TY_ERROR:
            return ERROR_T
        self.error(
            "cannot access field '" + expr.field + "' on " + type_name(obj_type),
            expr.pos,
        )
        return None

    def _union_field_type(self, union: UnionT, field: str) -> Type | None:
        """If all members are structs with the same-typed field, return it.

        Unions containing nil (optionals) are excluded — field access
        requires narrowing first.
        """
        for m in union.members:
            if type_eq(m, NIL_T):
                return None
        result: Type | None = None
        for m in union.members:
            if not isinstance(m, StructT):
                return None
            if field not in m.fields:
                return None
            ft = m.fields[field]
            if result is None:
                result = ft
            else:
                if not type_eq(result, ft):
                    return None
        return result

    def _interface_field_type(self, iface: InterfaceT, field: str) -> Type | None:
        """Return field type if ALL struct variants share it with compatible types."""
        result: Type | None = None
        for vname in iface.variants:
            vtype = self.types.get(vname)
            if vtype is None or not isinstance(vtype, StructT):
                return None
            if field not in vtype.fields:
                return None
            ft = vtype.fields[field]
            if result is None:
                result = ft
            else:
                if not type_eq(result, ft):
                    return None
        return result

    def check_tuple_access(self, expr: TTupleAccess) -> Type | None:
        obj_type = self.check_expr(expr.obj, None)
        if obj_type is None:
            return None
        if not isinstance(obj_type, TupleT):
            self.error(
                "tuple access on non-tuple type " + type_name(obj_type), expr.pos
            )
            return None
        if expr.index < 0 or expr.index >= len(obj_type.elements):
            self.error(
                "tuple index "
                + str(expr.index)
                + " out of range for "
                + type_name(obj_type),
                expr.pos,
            )
            return None
        return obj_type.elements[expr.index]

    def check_index(self, expr: TIndex) -> Type | None:
        obj_type = self.check_expr(expr.obj, None)
        if obj_type is None:
            return None
        if obj_type.kind == TY_ERROR:
            return ERROR_T
        idx_type = self.check_expr(expr.index, None)
        if isinstance(obj_type, ListT):
            if (
                idx_type is not None
                and idx_type.kind != TY_ERROR
                and not type_eq(idx_type, INT_T)
            ):
                self.error(
                    "list index must be int, got " + type_name(idx_type), expr.pos
                )
            return obj_type.element
        if type_eq(obj_type, STRING_T):
            if (
                idx_type is not None
                and idx_type.kind != TY_ERROR
                and not type_eq(idx_type, INT_T)
            ):
                self.error(
                    "string index must be int, got " + type_name(idx_type), expr.pos
                )
            return RUNE_T
        if type_eq(obj_type, BYTES_T):
            if (
                idx_type is not None
                and idx_type.kind != TY_ERROR
                and not type_eq(idx_type, INT_T)
            ):
                self.error(
                    "bytes index must be int, got " + type_name(idx_type), expr.pos
                )
            return BYTE_T
        if isinstance(obj_type, MapT):
            if idx_type is not None and not is_assignable(idx_type, obj_type.key):
                # Allow numeric interchangeability for map keys
                idx_numeric = (
                    idx_type.kind == TY_INT
                    or idx_type.kind == TY_FLOAT
                    or idx_type.kind == TY_BOOL
                )
                key_numeric = (
                    obj_type.key.kind == TY_INT
                    or obj_type.key.kind == TY_FLOAT
                    or obj_type.key.kind == TY_BOOL
                )
                if not (idx_numeric and key_numeric):
                    self.error(
                        "map key must be "
                        + type_name(obj_type.key)
                        + ", got "
                        + type_name(idx_type),
                        expr.pos,
                    )
            return obj_type.value
        if isinstance(obj_type, UnionT):
            self.error("cannot index union", expr.pos)
            return ERROR_T
        self.error("cannot index " + type_name(obj_type), expr.pos)
        return None

    def check_slice(self, expr: TSlice) -> Type | None:
        obj_type = self.check_expr(expr.obj, None)
        if obj_type is None:
            return None
        if obj_type.kind == TY_ERROR:
            return ERROR_T
        low_type = self.check_expr(expr.low, INT_T)
        high_type = self.check_expr(expr.high, INT_T)
        if (
            low_type is not None
            and low_type.kind != TY_ERROR
            and not type_eq(low_type, INT_T)
        ):
            self.error("slice bound must be int, got " + type_name(low_type), expr.pos)
        if (
            high_type is not None
            and high_type.kind != TY_ERROR
            and not type_eq(high_type, INT_T)
        ):
            self.error("slice bound must be int, got " + type_name(high_type), expr.pos)
        if isinstance(obj_type, ListT):
            return obj_type
        if isinstance(obj_type, TupleT) and obj_type.elements:
            return ListT(kind="list", element=obj_type.elements[0])
        if type_eq(obj_type, STRING_T):
            return STRING_T
        if type_eq(obj_type, BYTES_T):
            return BYTES_T
        self.error("cannot slice " + type_name(obj_type), expr.pos)
        return None

    def check_call(self, expr: TCall, expected: Type | None) -> Type | None:
        # Struct/interface constructor and builtin dispatch
        if isinstance(expr.func, TVar):
            func_name = expr.func.name
            resolved = self._try_lookup(func_name)
            if resolved is not None and isinstance(resolved, StructT):
                return self.check_struct_constructor(resolved, expr.args, expr.pos)
            if resolved is not None and isinstance(resolved, InterfaceT):
                # Find common fields across all variants (base class fields)
                common: dict[str, Type] | None = None
                for vname in resolved.variants:
                    vtype = self.types.get(vname)
                    if vtype is not None and isinstance(vtype, StructT):
                        if common is None:
                            common = vtype.fields.copy()
                        else:
                            keep: dict[str, Type] = {}
                            for k in common:
                                if k in vtype.fields:
                                    keep[k] = common[k]
                            common = keep
                if common is not None and len(expr.args) <= len(common):
                    for a in expr.args:
                        aname = a.name
                        if aname is None:
                            self.check_expr(a.value, None)
                        elif aname in common:
                            at = self.check_expr(a.value, common[aname])
                            if at is not None and not is_assignable(at, common[aname]):
                                self.error(
                                    "field '"
                                    + aname
                                    + "': cannot assign "
                                    + type_name(at)
                                    + " to "
                                    + type_name(common[aname]),
                                    a.pos,
                                )
                        else:
                            self.error(
                                "'" + func_name + "' has no field '" + aname + "'",
                                a.pos,
                            )
                    return resolved
                self.error("cannot call " + type_name(resolved), expr.pos)
                return None
            if resolved is not None and isinstance(resolved, FnT):
                pnames = self.fn_param_names.get(func_name)
                return self.check_fn_call(resolved, expr.args, expr.pos, pnames)
            # Builtin functions (after struct/interface/fn resolution)
            if func_name in BUILTIN_NAMES and func_name not in self.functions:
                return self.check_builtin_call(func_name, expr.args, expr.pos, expected)
            if resolved is not None:
                if resolved.kind == TY_ERROR:
                    return ERROR_T
                self.error("cannot call " + type_name(resolved), expr.pos)
                return None
            return None
        # Method call: TCall with TFieldAccess func
        if isinstance(expr.func, TFieldAccess):
            return self.check_method_call(expr.func, expr.args, expr.pos)
        # Function value call
        func_type = self.check_expr(expr.func, None)
        if func_type is None:
            return None
        if isinstance(func_type, FnT):
            return self.check_fn_call(func_type, expr.args, expr.pos)
        if func_type.kind == TY_ERROR:
            return ERROR_T
        self.error("cannot call " + type_name(func_type), expr.pos)
        return None

    def check_fn_call(
        self,
        fn: FnT,
        args: list[TArg],
        pos: Pos,
        param_names: list[str] | None = None,
    ) -> Type | None:
        min_p = fn.min_params if fn.min_params >= 0 else len(fn.params)
        if len(args) < min_p or len(args) > len(fn.params):
            self.error(
                "expected " + str(len(fn.params)) + " arguments, got " + str(len(args)),
                pos,
            )
            return fn.ret
        # Check for named args
        has_named = len(args) > 0 and args[0].name is not None
        if has_named and param_names is not None:
            # Validate no mixing
            for a in args:
                if a.name is None:
                    self.error("cannot mix positional and named arguments", a.pos)
                    return fn.ret
            # Resolve named args to positional
            for a in args:
                assert a.name is not None
                if a.name not in param_names:
                    self.error("no parameter '" + a.name + "'", a.pos)
                    continue
                idx = param_names.index(a.name)
                arg_type = self.check_expr(a.value, fn.params[idx])
                if arg_type is not None and not is_assignable(arg_type, fn.params[idx]):
                    self.error(
                        "argument '"
                        + a.name
                        + "': cannot pass "
                        + type_name(arg_type)
                        + " as "
                        + type_name(fn.params[idx]),
                        a.pos,
                    )
        elif has_named:
            # Named args but no param names available — reject
            self.error("named arguments not supported for fn values", pos)
            # Just check types positionally
            i = 0
            while i < len(args):
                if i < len(fn.params):
                    arg_type = self.check_expr(args[i].value, fn.params[i])
                    if arg_type is not None and not is_assignable(
                        arg_type, fn.params[i]
                    ):
                        self.error(
                            "argument "
                            + str(i + 1)
                            + ": cannot pass "
                            + type_name(arg_type)
                            + " as "
                            + type_name(fn.params[i]),
                            args[i].pos,
                        )
                i += 1
        else:
            # Check for positional args mixed with named
            i = 0
            while i < len(args):
                if args[i].name is not None:
                    self.error("cannot mix positional and named arguments", args[i].pos)
                    return fn.ret
                if i < len(fn.params):
                    arg_type = self.check_expr(args[i].value, fn.params[i])
                    if arg_type is not None and not is_assignable(
                        arg_type, fn.params[i]
                    ):
                        self.error(
                            "argument "
                            + str(i + 1)
                            + ": cannot pass "
                            + type_name(arg_type)
                            + " as "
                            + type_name(fn.params[i]),
                            args[i].pos,
                        )
                i += 1
        return fn.ret

    def check_struct_constructor(
        self, st: StructT, args: list[TArg], pos: Pos
    ) -> Type | None:
        field_names = st.field_order if st.field_order else list(st.fields.keys())
        if len(args) == 0 and len(field_names) == 0:
            return st
        min_f = st.min_fields if st.min_fields >= 0 else len(field_names)
        if len(args) < min_f or len(args) > len(field_names):
            self.error(
                st.name
                + " has "
                + str(len(field_names))
                + " fields, got "
                + str(len(args))
                + " arguments",
                pos,
            )
            return st
        if len(args) == 0:
            return st
        # Check if named or positional
        if args[0].name is not None:
            # Named construction — check for duplicate field names
            seen_names: set[str] = set()
            for arg in args:
                if arg.name is None:
                    self.error("cannot mix positional and named arguments", arg.pos)
                    return st
                if arg.name in seen_names:
                    self.error("duplicate field '" + arg.name + "'", arg.pos)
                    continue
                seen_names.add(arg.name)
                if arg.name not in st.fields:
                    self.error(
                        "'" + st.name + "' has no field '" + arg.name + "'", arg.pos
                    )
                    continue
                expected_type = st.fields[arg.name]
                arg_type = self.check_expr(arg.value, expected_type)
                if arg_type is not None and not is_assignable(arg_type, expected_type):
                    self.error(
                        "field '"
                        + arg.name
                        + "': cannot assign "
                        + type_name(arg_type)
                        + " to "
                        + type_name(expected_type),
                        arg.pos,
                    )
        else:
            # Positional construction
            i = 0
            while i < len(args):
                if args[i].name is not None:
                    self.error("cannot mix positional and named arguments", args[i].pos)
                    return st
                expected_type2 = st.fields[field_names[i]]
                arg_type2 = self.check_expr(args[i].value, expected_type2)
                if arg_type2 is not None and not is_assignable(
                    arg_type2, expected_type2
                ):
                    self.error(
                        "field '"
                        + field_names[i]
                        + "': cannot assign "
                        + type_name(arg_type2)
                        + " to "
                        + type_name(expected_type2),
                        args[i].pos,
                    )
                i += 1
        return st

    def check_method_call(
        self, access: TFieldAccess, args: list[TArg], pos: Pos
    ) -> Type | None:
        # Check for enum access used as call (shouldn't happen, but guard)
        if isinstance(access.obj, TVar):
            resolved = self.lookup(access.obj.name, access.obj.pos)
            if resolved is not None and isinstance(resolved, EnumT):
                self.error("enum variant is not callable", pos)
                return None
        obj_type = self.check_expr(access.obj, None)
        if obj_type is None:
            return None
        if contains_nil(obj_type) and not isinstance(obj_type, StructT):
            self.error("cannot call method on optional; narrow first", pos)
            return None
        if isinstance(obj_type, StructT):
            if access.field in obj_type.methods:
                method = obj_type.methods[access.field]
                return self.check_fn_call(method, args, pos)
            if access.field in obj_type.fields:
                field_type = obj_type.fields[access.field]
                if isinstance(field_type, FnT):
                    return self.check_fn_call(field_type, args, pos)
                self.error(
                    "'" + access.field + "' is not a method of " + obj_type.name, pos
                )
                return None
            self.error(
                "'" + obj_type.name + "' has no method '" + access.field + "'", pos
            )
            return None
        if isinstance(obj_type, InterfaceT):
            return ERROR_T
        if isinstance(obj_type, (MapT, ListT, SetT)):
            return self._check_collection_method(obj_type, access.field, args, pos)
        if obj_type.kind in (TY_STRING, TY_BYTES, TY_ERROR):
            return ERROR_T
        self.error("cannot call method on " + type_name(obj_type), pos)
        return None

    def _check_collection_method(
        self, obj_type: Type, method: str, args: list[TArg], pos: Pos
    ) -> Type | None:
        if isinstance(obj_type, MapT):
            if method == "get":
                return obj_type.value
            if method == "keys":
                return ListT(kind="list", element=obj_type.key)
            if method == "values":
                return ListT(kind="list", element=obj_type.value)
            if method == "items":
                return ListT(
                    kind="list",
                    element=TupleT(
                        kind="tuple", elements=[obj_type.key, obj_type.value]
                    ),
                )
            if method in ("pop", "setdefault"):
                return obj_type.value
        if isinstance(obj_type, ListT):
            if method == "append":
                return VOID_T
            if method == "pop":
                return obj_type.element
            if method in ("extend", "insert", "remove", "clear", "reverse", "sort"):
                return VOID_T
        if isinstance(obj_type, SetT):
            if method in ("add", "remove", "discard", "clear"):
                return VOID_T
        return ERROR_T

    def check_list_lit(self, expr: TListLit, expected: Type | None) -> Type | None:
        if len(expr.elements) == 0:
            if expected is not None and isinstance(expected, ListT):
                return expected
            return ListT(kind="list", element=ERROR_T)
        # Use expected element type if available
        elem_expected: Type | None = None
        if expected is not None and isinstance(expected, ListT):
            elem_expected = expected.element
        first = self.check_expr(expr.elements[0], elem_expected)
        if first is None:
            return None
        check_type = elem_expected if elem_expected is not None else first
        if not is_assignable(first, check_type):
            self.error(
                "list element: cannot assign "
                + type_name(first)
                + " to "
                + type_name(check_type),
                expr.elements[0].pos,
            )
        i = 1
        while i < len(expr.elements):
            elem = self.check_expr(expr.elements[i], check_type)
            if elem is not None and not is_assignable(elem, check_type):
                self.error(
                    "list elements must have same type, got "
                    + type_name(check_type)
                    + " and "
                    + type_name(elem),
                    expr.elements[i].pos,
                )
            i += 1
        return ListT(kind="list", element=check_type)

    def check_map_lit(self, expr: TMapLit, expected: Type | None) -> Type | None:
        if len(expr.entries) == 0:
            if expected is not None and isinstance(expected, MapT):
                return expected
            self.error("cannot infer type of empty map literal", expr.pos)
            return None
        key_expected: Type | None = None
        val_expected: Type | None = None
        if expected is not None and isinstance(expected, MapT):
            key_expected = expected.key
            val_expected = expected.value
        key_type = self.check_expr(expr.entries[0][0], key_expected)
        val_type = self.check_expr(expr.entries[0][1], val_expected)
        if key_type is None or val_type is None:
            return None
        check_key = key_expected if key_expected is not None else key_type
        check_val = val_expected if val_expected is not None else val_type
        # Track literal keys for duplicate detection
        seen_keys: list[str] = []
        k0_val = _literal_key_value(expr.entries[0][0])
        if k0_val is not None:
            seen_keys.append(k0_val)
        i = 1
        while i < len(expr.entries):
            ki, vi = expr.entries[i]
            ki_val = _literal_key_value(ki)
            if ki_val is not None:
                if ki_val in seen_keys:
                    self.error("duplicate key in map literal", ki.pos)
                else:
                    seen_keys.append(ki_val)
            kt = self.check_expr(ki, check_key)
            vt = self.check_expr(vi, check_val)
            if kt is not None and not is_assignable(kt, check_key):
                kt_compat = kt.kind == TY_BOOL or kt.kind == TY_INT
                ck_compat = check_key.kind == TY_BOOL or check_key.kind == TY_INT
                if not (kt_compat and ck_compat):
                    self.error("map keys must have same type", ki.pos)
            if vt is not None and not is_assignable(vt, check_val):
                widened = self._widen_to_common_interface(check_val, vt)
                if widened is not None:
                    check_val = widened
                else:
                    self.error("map values must have same type", vi.pos)
            i += 1
        return MapT(kind="map", key=check_key, value=check_val)

    def _widen_to_common_interface(self, a: Type, b: Type) -> Type | None:
        if is_assignable(a, b):
            return b
        if is_assignable(b, a):
            return a
        for t in self.types.values():
            if isinstance(t, InterfaceT):
                if is_assignable(a, t) and is_assignable(b, t):
                    return t
        return None

    def check_set_lit(self, expr: TSetLit, expected: Type | None) -> Type | None:
        if len(expr.elements) == 0:
            if expected is not None and isinstance(expected, SetT):
                return expected
            self.error("cannot infer type of empty set literal", expr.pos)
            return None
        first = self.check_expr(expr.elements[0], None)
        if first is None:
            return None
        i = 1
        while i < len(expr.elements):
            elem = self.check_expr(expr.elements[i], first)
            if elem is not None and not type_eq(elem, first):
                self.error("set elements must have same type", expr.elements[i].pos)
            i += 1
        return SetT(kind="set", element=first)

    def check_tuple_lit(self, expr: TTupleLit, expected: Type | None) -> Type | None:
        elem_types: list[Type] = []
        i = 0
        while i < len(expr.elements):
            exp_elem: Type | None = None
            if (
                expected is not None
                and isinstance(expected, TupleT)
                and i < len(expected.elements)
            ):
                exp_elem = expected.elements[i]
            et = self.check_expr(expr.elements[i], exp_elem)
            if et is None:
                return None
            elem_types.append(et)
            i += 1
        return TupleT(kind="tuple", elements=elem_types)

    def check_fn_lit(self, expr: TFnLit, expected: Type | None) -> Type | None:
        params: list[Type] = []
        for p in expr.params:
            if p.typ is not None:
                params.append(self.resolve_type(p.typ))
            else:
                self.error("fn literal parameter must have a type", p.pos)
                params.append(ERROR_T)
        ret = self.resolve_type(expr.ret)
        # Check for captured variables (no closures allowed)
        param_names: set[str] = set()
        for p in expr.params:
            param_names.add(p.name)
        # Check body
        old_ret = self.current_fn_ret
        self.current_fn_ret = ret
        saved_uninit = set(self.uninitialized)
        self.uninitialized = set()
        self.enter_scope()
        for p in expr.params:
            if p.typ is not None:
                pt = self.resolve_type(p.typ)
                self.declare(p.name, pt, p.pos)
        self.check_closure_captures(expr.body, param_names, expr.pos)
        is_arrow = expr.annotations.get("fn_lit.arrow") == "true"
        first = expr.body[0] if expr.body else None
        if is_arrow and isinstance(first, TExprStmt):
            arrow_type = self.check_expr(first.expr, ret)
            if arrow_type is not None and not type_eq(ret, VOID_T):
                if not is_assignable(arrow_type, ret):
                    self.error(
                        "cannot assign "
                        + type_name(arrow_type)
                        + " to "
                        + type_name(ret),
                        expr.body[0].pos,
                    )
        else:
            self.check_stmts(expr.body)
            if not type_eq(ret, VOID_T) and not _block_is_complete(expr.body):
                self.error("not all paths return a value", expr.pos)
        self.exit_scope()
        self.current_fn_ret = old_ret
        self.uninitialized = saved_uninit
        return FnT(kind="fn", params=params, ret=ret)

    def check_closure_captures(
        self, stmts: list[TStmt], param_names: set[str], pos: Pos
    ) -> None:
        """Check that fn literal body doesn't capture variables from enclosing scope."""
        local_names = set(param_names)
        for s in stmts:
            self._scan_stmt_for_captures(s, local_names, pos)

    def check_closure_captures_expr(
        self, expr: TExpr, param_names: set[str], pos: Pos
    ) -> None:
        self._scan_expr_for_captures(expr, param_names, pos)

    def _scan_expr_for_captures(
        self, expr: TExpr, param_names: set[str], pos: Pos
    ) -> None:
        if isinstance(expr, TVar):
            name = expr.name
            if name in param_names:
                return
            if name in self.functions:
                return
            if name in self.types:
                return
            if name in BUILTIN_NAMES:
                return
            # This is a capture
            self.error("cannot capture '" + name + "' in fn literal", expr.pos)
            return
        if isinstance(expr, TBinaryOp):
            self._scan_expr_for_captures(expr.left, param_names, pos)
            self._scan_expr_for_captures(expr.right, param_names, pos)
        elif isinstance(expr, TUnaryOp):
            self._scan_expr_for_captures(expr.operand, param_names, pos)
        elif isinstance(expr, TTernary):
            self._scan_expr_for_captures(expr.cond, param_names, pos)
            self._scan_expr_for_captures(expr.then_expr, param_names, pos)
            self._scan_expr_for_captures(expr.else_expr, param_names, pos)
        elif isinstance(expr, TFieldAccess):
            self._scan_expr_for_captures(expr.obj, param_names, pos)
        elif isinstance(expr, TTupleAccess):
            self._scan_expr_for_captures(expr.obj, param_names, pos)
        elif isinstance(expr, TIndex):
            self._scan_expr_for_captures(expr.obj, param_names, pos)
            self._scan_expr_for_captures(expr.index, param_names, pos)
        elif isinstance(expr, TSlice):
            self._scan_expr_for_captures(expr.obj, param_names, pos)
            self._scan_expr_for_captures(expr.low, param_names, pos)
            self._scan_expr_for_captures(expr.high, param_names, pos)
        elif isinstance(expr, TCall):
            self._scan_expr_for_captures(expr.func, param_names, pos)
            for a in expr.args:
                self._scan_expr_for_captures(a.value, param_names, pos)
        elif isinstance(expr, TListLit):
            for e in expr.elements:
                self._scan_expr_for_captures(e, param_names, pos)
        elif isinstance(expr, TMapLit):
            for k, v in expr.entries:
                self._scan_expr_for_captures(k, param_names, pos)
                self._scan_expr_for_captures(v, param_names, pos)
        elif isinstance(expr, TSetLit):
            for e in expr.elements:
                self._scan_expr_for_captures(e, param_names, pos)
        elif isinstance(expr, TTupleLit):
            for e in expr.elements:
                self._scan_expr_for_captures(e, param_names, pos)
        elif isinstance(expr, TFnLit):
            # Nested fn lits — the inner one can reference the outer's params
            inner_params: set[str] = set(param_names)
            for p in expr.params:
                inner_params.add(p.name)
            for s in expr.body:
                self._scan_stmt_for_captures(s, inner_params, pos)

    def _scan_stmt_for_captures(
        self, stmt: TStmt, param_names: set[str], pos: Pos
    ) -> None:
        if isinstance(stmt, TLetStmt):
            if stmt.value is not None:
                self._scan_expr_for_captures(stmt.value, param_names, pos)
            # The declared name becomes a local, not a capture
            param_names.add(stmt.name)
        elif isinstance(stmt, TAssignStmt):
            self._scan_expr_for_captures(stmt.target, param_names, pos)
            self._scan_expr_for_captures(stmt.value, param_names, pos)
        elif isinstance(stmt, TOpAssignStmt):
            self._scan_expr_for_captures(stmt.target, param_names, pos)
            self._scan_expr_for_captures(stmt.value, param_names, pos)
        elif isinstance(stmt, TTupleAssignStmt):
            for t in stmt.targets:
                self._scan_expr_for_captures(t, param_names, pos)
            self._scan_expr_for_captures(stmt.value, param_names, pos)
        elif isinstance(stmt, TReturnStmt):
            if stmt.value is not None:
                self._scan_expr_for_captures(stmt.value, param_names, pos)
        elif isinstance(stmt, TThrowStmt):
            self._scan_expr_for_captures(stmt.expr, param_names, pos)
        elif isinstance(stmt, TExprStmt):
            self._scan_expr_for_captures(stmt.expr, param_names, pos)
        elif isinstance(stmt, TIfStmt):
            self._scan_expr_for_captures(stmt.cond, param_names, pos)
            for s in stmt.then_body:
                self._scan_stmt_for_captures(s, param_names, pos)
            if stmt.else_body is not None:
                for s in stmt.else_body:
                    self._scan_stmt_for_captures(s, param_names, pos)
        elif isinstance(stmt, TWhileStmt):
            self._scan_expr_for_captures(stmt.cond, param_names, pos)
            for s in stmt.body:
                self._scan_stmt_for_captures(s, param_names, pos)
        elif isinstance(stmt, TForStmt):
            param_names = set(param_names)
            for b in stmt.binding:
                param_names.add(b)
            if isinstance(stmt.iterable, TRange):
                for a in stmt.iterable.args:
                    self._scan_expr_for_captures(a, param_names, pos)
            else:
                self._scan_expr_for_captures(stmt.iterable, param_names, pos)
            for s in stmt.body:
                self._scan_stmt_for_captures(s, param_names, pos)
        elif isinstance(stmt, TMatchStmt):
            self._scan_expr_for_captures(stmt.expr, param_names, pos)
            for case in stmt.cases:
                inner = set(param_names)
                if isinstance(case.pattern, TPatternType):
                    inner.add(case.pattern.name)
                for s in case.body:
                    self._scan_stmt_for_captures(s, inner, pos)
            if stmt.default is not None:
                inner2 = set(param_names)
                if stmt.default.name is not None:
                    inner2.add(stmt.default.name)
                for s in stmt.default.body:
                    self._scan_stmt_for_captures(s, inner2, pos)
        elif isinstance(stmt, TTryStmt):
            for s in stmt.body:
                self._scan_stmt_for_captures(s, param_names, pos)
            for catch in stmt.catches:
                inner3 = set(param_names)
                inner3.add(catch.name)
                for s in catch.body:
                    self._scan_stmt_for_captures(s, inner3, pos)
            if stmt.finally_body is not None:
                for s in stmt.finally_body:
                    self._scan_stmt_for_captures(s, param_names, pos)

    # ── Built-in function checking ────────────────────────────

    def check_builtin_call(
        self, name: str, args: list[TArg], pos: Pos, expected: Type | None
    ) -> Type | None:
        for a in args:
            if a.name is not None:
                self.error("named arguments not supported for built-in " + name, pos)
                break
        arg_types: list[Type | None] = []
        for a in args:
            arg_types.append(self.check_expr(a.value, None))
        n = len(args)
        ctx = _BuiltinCtx(self, name, arg_types, len(arg_types), pos)

        # ── Numeric ──
        if name == "Abs":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and t.kind not in (TY_INT, TY_FLOAT):
                self.error("Abs requires int or float", pos)
            return t
        if name == "Min" or name == "Max":
            if n == 1:
                t = _bctx_arg(ctx, 0)
                if t is not None and isinstance(t, ListT):
                    return t.element
                if t is not None:
                    self.error(name + " with 1 argument requires list", pos)
                return None
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and t2 is not None:
                if not type_eq(t1, t2):
                    self.error(
                        name
                        + " requires same type, got "
                        + type_name(t1)
                        + " and "
                        + type_name(t2),
                        pos,
                    )
                if t1.kind not in (TY_INT, TY_FLOAT, TY_BYTE):
                    self.error(name + " requires int, float, or byte", pos)
            return t1
        if name == "Sum":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None:
                if isinstance(t, ListT) and t.element.kind in (TY_INT, TY_FLOAT):
                    return t.element
                if isinstance(t, ListT) and t.element.kind == TY_ERROR:
                    return INT_T
                if isinstance(t, SetT) and t.element.kind in (TY_INT, TY_FLOAT):
                    return t.element
                if isinstance(t, TupleT) and len(t.elements) > 0:
                    elem = t.elements[0]
                    if elem.kind in (TY_INT, TY_FLOAT):
                        return elem
                if isinstance(t, TupleT) and len(t.elements) == 0:
                    return INT_T
                self.error(
                    "Sum requires list[int], list[float], set[int], or set[float]", pos
                )
            return None
        if name == "Pow":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and t2 is not None:
                if not type_eq(t1, t2):
                    self.error("Pow requires same type", pos)
                if t1.kind not in (TY_INT, TY_FLOAT):
                    self.error("Pow requires int or float", pos)
            return t1
        if name == "FloorDiv" or name == "PythonMod":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None:
                t1_fd = _unwrap_nil_union(t1)
            else:
                t1_fd = None
            if (
                t1_fd is not None
                and t1_fd.kind != TY_ERROR
                and t1_fd.kind not in (TY_INT, TY_FLOAT)
            ):
                self.error(name + " requires int or float", pos)
            return t1_fd
        if name == "ReplaceSlice":
            if not _bctx_require(ctx, 4):
                return None
            t1 = _bctx_arg(ctx, 0)
            if t1 is not None and not isinstance(t1, ListT):
                self.error("ReplaceSlice requires list as first argument", pos)
            return VOID_T
        if name in ("Floor", "Ceil"):
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, FLOAT_T):
                self.error(name + " requires float", pos)
            return INT_T
        if name == "Round":
            if not _bctx_require_range(ctx, 1, 2):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, FLOAT_T):
                self.error("Round requires float", pos)
            if ctx.n == 2:
                t2 = _bctx_arg(ctx, 1)
                if t2 is not None and not type_eq(t2, INT_T):
                    self.error("Round ndigits must be int", pos)
                return FLOAT_T
            return INT_T
        if name == "DivMod":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and not type_eq(t1, INT_T):
                self.error("DivMod requires int", pos)
            if t2 is not None and not type_eq(t2, INT_T):
                self.error("DivMod requires int", pos)
            return TupleT(kind="tuple", elements=[INT_T, INT_T])
        if name in ("WrappingAdd", "WrappingSub", "WrappingMul"):
            if not self.strict_math:
                self.error(name + " requires strict_math mode", pos)
                return None
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and not type_eq(t1, INT_T):
                self.error(name + " requires int", pos)
            if t2 is not None and not type_eq(t2, INT_T):
                self.error(name + " requires int", pos)
            return INT_T

        # ── Bytes ──
        if name == "Encode":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, STRING_T):
                self.error("Encode requires string", pos)
            return BYTES_T
        if name == "Decode":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, BYTES_T):
                self.error("Decode requires bytes", pos)
            return STRING_T

        # ── Len ──
        if name == "Len":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and t.kind != TY_ERROR:
                t_inner = _unwrap_nil_union(t)
                if not (
                    isinstance(t_inner, (ListT, MapT, SetT))
                    or t_inner.kind in (TY_STRING, TY_BYTES)
                ):
                    self.error("Len requires string, bytes, list, map, or set", pos)
            return INT_T

        # ── Concat ──
        if name == "Concat":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and t2 is not None:
                if t1.kind == TY_ERROR or t2.kind == TY_ERROR:
                    return ERROR_T
                t1u = _unwrap_nil_union(t1)
                t2u = _unwrap_nil_union(t2)
                if type_eq(t1u, STRING_T) and type_eq(t2u, STRING_T):
                    return STRING_T
                if type_eq(t1u, BYTES_T) and type_eq(t2u, BYTES_T):
                    return BYTES_T
                if isinstance(t1u, ListT) and isinstance(t2u, ListT):
                    return t1u
                if isinstance(t1u, (ListT, TupleT)) and isinstance(
                    t2u, (ListT, TupleT)
                ):
                    if isinstance(t1u, ListT):
                        return t1u
                    if isinstance(t2u, ListT):
                        return t2u
                    if isinstance(t1u, TupleT):
                        return ListT(
                            kind="list",
                            element=t1u.elements[0] if t1u.elements else ERROR_T,
                        )
                self.error("Concat requires two strings, two bytes, or two lists", pos)
            return STRING_T

        # ── Append ──
        if name == "Append":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None:
                if not isinstance(t1, ListT):
                    self.error("Append requires list as first argument", pos)
                elif t2 is not None and not is_assignable(t2, t1.element):
                    # Allow byte↔int for list operations (Python bytes iterate as int)
                    t2_ok = t2.kind == TY_BYTE or t2.kind == TY_INT
                    e_ok = t1.element.kind == TY_BYTE or t1.element.kind == TY_INT
                    if not (t2_ok and e_ok):
                        self.error(
                            "cannot append " + type_name(t2) + " to " + type_name(t1),
                            pos,
                        )
            return VOID_T

        # ── Insert ──
        if name == "Insert":
            if not _bctx_require(ctx, 3):
                return None
            t1 = _bctx_arg(ctx, 0)
            if t1 is not None and not isinstance(t1, ListT):
                self.error("Insert requires list as first argument", pos)
            t2 = _bctx_arg(ctx, 1)
            if t2 is not None and not type_eq(t2, INT_T):
                self.error("Insert index must be int", pos)
            t3 = _bctx_arg(ctx, 2)
            if t1 is not None and isinstance(t1, ListT) and t3 is not None:
                if not is_assignable(t3, t1.element):
                    self.error(
                        "cannot pass " + type_name(t3) + " as " + type_name(t1.element),
                        pos,
                    )
            return VOID_T

        # ── Pop ──
        if name == "Pop":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and isinstance(t, ListT):
                return t.element
            if t is not None and isinstance(t, SetT):
                return t.element
            if t is not None:
                self.error("Pop requires list or set", pos)
            return None

        # ── RemoveAt ──
        if name == "RemoveAt":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            if t1 is not None and not isinstance(t1, ListT):
                self.error("RemoveAt requires list", pos)
            t2 = _bctx_arg(ctx, 1)
            if t2 is not None and not type_eq(t2, INT_T):
                self.error("RemoveAt index must be int", pos)
            return VOID_T

        # ── IndexOf ──
        if name == "IndexOf":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and not isinstance(t1, ListT):
                self.error("IndexOf requires list", pos)
            elif isinstance(t1, ListT) and t2 is not None:
                if not is_assignable(t2, t1.element):
                    self.error(
                        "cannot pass " + type_name(t2) + " as " + type_name(t1.element),
                        pos,
                    )
            return INT_T

        # ── Contains ──
        if name == "Contains":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and t1.kind != TY_ERROR:
                t1u = _unwrap_nil_union(t1)
                if isinstance(t1u, ListT):
                    if t2 is not None and not is_assignable(t2, t1u.element):
                        self.error(
                            "cannot pass "
                            + type_name(t2)
                            + " as "
                            + type_name(t1u.element),
                            pos,
                        )
                elif isinstance(t1u, SetT):
                    if t2 is not None and not is_assignable(t2, t1u.element):
                        self.error(
                            "cannot pass "
                            + type_name(t2)
                            + " as "
                            + type_name(t1u.element),
                            pos,
                        )
                elif isinstance(t1u, MapT):
                    if t2 is not None and not is_assignable(t2, t1u.key):
                        self.error(
                            "cannot pass "
                            + type_name(t2)
                            + " as "
                            + type_name(t1u.key),
                            pos,
                        )
                elif type_eq(t1u, STRING_T):
                    pass
                elif type_eq(t1u, BYTES_T):
                    pass
                elif isinstance(t1u, TupleT):
                    pass
                else:
                    self.error(
                        "Contains requires list, set, map, string, or bytes", pos
                    )
            return BOOL_T

        # ── Get ──
        if name == "Get":
            if not _bctx_require_range(ctx, 2, 3):
                return None
            t1 = _bctx_arg(ctx, 0)
            if t1 is not None and not isinstance(t1, MapT):
                self.error("Get requires map as first argument", pos)
                return None
            if t1 is not None and isinstance(t1, MapT):
                if n == 3:
                    t3 = _bctx_arg(ctx, 2)
                    if t3 is not None and not is_assignable(t3, t1.value):
                        self.error(
                            "default value: cannot assign "
                            + type_name(t3)
                            + " to "
                            + type_name(t1.value),
                            pos,
                        )
                    return t1.value
                return make_optional(t1.value)
            return None

        # ── Delete ──
        if name == "Delete":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and not isinstance(t1, MapT):
                self.error("Delete requires map", pos)
            elif isinstance(t1, MapT) and t2 is not None:
                if not is_assignable(t2, t1.key):
                    self.error(
                        "cannot pass " + type_name(t2) + " as " + type_name(t1.key),
                        pos,
                    )
            return VOID_T

        # ── Keys / Values / Items ──
        if name == "Keys":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and isinstance(t, MapT):
                return ListT(kind="list", element=t.key)
            if t is not None:
                self.error("Keys requires map", pos)
            return None
        if name == "Values":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and isinstance(t, MapT):
                return ListT(kind="list", element=t.value)
            if t is not None:
                self.error("Values requires map", pos)
            return None
        if name == "Items":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and isinstance(t, MapT):
                return ListT(
                    kind="list", element=TupleT(kind="tuple", elements=[t.key, t.value])
                )
            if t is not None:
                self.error("Items requires map", pos)
            return None

        # ── Merge ──
        if name == "Merge":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and not isinstance(t1, MapT):
                self.error("Merge requires map", pos)
                return None
            if t2 is not None and not isinstance(t2, MapT):
                self.error("Merge requires map", pos)
                return None
            if (
                t1 is not None
                and t2 is not None
                and isinstance(t1, MapT)
                and isinstance(t2, MapT)
            ):
                if not type_eq(t1, t2):
                    # Allow merge with empty map
                    if t1.key.kind != TY_ERROR and t2.key.kind != TY_ERROR:
                        self.error(
                            "Merge maps must be same type, got "
                            + type_name(t1)
                            + " and "
                            + type_name(t2),
                            pos,
                        )
                return t1
            if t1 is not None and isinstance(t1, MapT):
                return t1
            return None

        # ── PopItem ──
        if name == "PopItem":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and isinstance(t, MapT):
                return TupleT(kind="tuple", elements=[t.key, t.value])
            if t is not None:
                self.error("PopItem requires map", pos)
            return None

        # ── MapFromKeys ──
        if name == "MapFromKeys":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and isinstance(t1, ListT):
                val_ty = t2 if t2 is not None else ERROR_T
                return MapT(kind="map", key=t1.element, value=val_ty)
            if t1 is not None:
                self.error("MapFromKeys requires list as first argument", pos)
            return None

        # ── Map() / Set() ──
        if name == "Map":
            if not _bctx_require(ctx, 0):
                return None
            if expected is not None and isinstance(expected, MapT):
                return expected
            return MapT(kind="map", key=ERROR_T, value=ERROR_T)
        if name == "Set":
            if not _bctx_require(ctx, 0):
                return None
            if expected is not None and isinstance(expected, SetT):
                return expected
            return SetT(kind="set", element=ERROR_T)

        # ── Add / Remove (set) ──
        if name == "Add":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and not isinstance(t1, SetT):
                self.error("Add requires set as first argument", pos)
            elif isinstance(t1, SetT) and t2 is not None:
                if not is_assignable(t2, t1.element):
                    self.error(
                        "cannot pass " + type_name(t2) + " as " + type_name(t1.element),
                        pos,
                    )
            return VOID_T
        if name == "Remove":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and not isinstance(t1, SetT):
                self.error("Remove requires set as first argument", pos)
            elif isinstance(t1, SetT) and t2 is not None:
                if not is_assignable(t2, t1.element):
                    self.error(
                        "cannot pass " + type_name(t2) + " as " + type_name(t1.element),
                        pos,
                    )
            return VOID_T

        # ── Union / Intersection / Difference ──
        if name in ("Union", "Intersection", "Difference"):
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and not isinstance(t1, SetT):
                self.error(name + " requires set as first argument", pos)
                return None
            if t2 is not None and not isinstance(t2, SetT):
                self.error(name + " requires set as second argument", pos)
                return None
            if (
                t1 is not None
                and t2 is not None
                and isinstance(t1, SetT)
                and isinstance(t2, SetT)
            ):
                if not type_eq(t1, t2):
                    if t1.element.kind != TY_ERROR and t2.element.kind != TY_ERROR:
                        self.error(
                            name
                            + " sets must be same type, got "
                            + type_name(t1)
                            + " and "
                            + type_name(t2),
                            pos,
                        )
                return t1
            if t1 is not None and isinstance(t1, SetT):
                return t1
            return None

        # ── Bytes / BytesFrom ──
        if name == "Bytes":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, INT_T):
                self.error("Bytes requires int", pos)
            return BYTES_T
        if name == "BytesFrom":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None:
                if not isinstance(t, ListT):
                    self.error("BytesFrom requires list", pos)
            return BYTES_T

        # ── RangeList ──
        if name == "RangeList":
            if not _bctx_require(ctx, 3):
                return None
            i = 0
            while i < 3:
                t = _bctx_arg(ctx, i)
                if t is not None and not type_eq(t, INT_T):
                    self.error("RangeList requires int arguments", pos)
                i += 1
            return ListT(kind="list", element=INT_T)

        # ── MapFromPairs ──
        if name == "MapFromPairs":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None:
                t_mfp = _unwrap_nil_union(t)
                if isinstance(t_mfp, ListT) and isinstance(t_mfp.element, TupleT):
                    elems = t_mfp.element.elements
                    if len(elems) == 2:
                        return MapT(kind="map", key=elems[0], value=elems[1])
                self.error("MapFromPairs requires list of 2-tuples", pos)
            return None

        # ── ListCompare ──
        if name == "ListCompare":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and not isinstance(t1, ListT):
                self.error("ListCompare requires list as first argument", pos)
            if t2 is not None and not isinstance(t2, ListT):
                self.error("ListCompare requires list as second argument", pos)
            return INT_T

        # ── Zip ──
        if name == "Zip":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None and not isinstance(t1, ListT):
                self.error("Zip requires list as first argument", pos)
                return None
            if t2 is not None and not isinstance(t2, ListT):
                self.error("Zip requires list as second argument", pos)
                return None
            if (
                t1 is not None
                and t2 is not None
                and isinstance(t1, ListT)
                and isinstance(t2, ListT)
            ):
                return ListT(
                    kind="list",
                    element=TupleT(kind="tuple", elements=[t1.element, t2.element]),
                )
            return None

        # ── SetFromList ──
        if name == "SetFromList":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None:
                tu = _unwrap_nil_union(t)
                if isinstance(tu, ListT):
                    return SetT(kind="set", element=tu.element)
                if isinstance(tu, SetT):
                    return tu
                if tu.kind != TY_ERROR:
                    self.error("SetFromList requires list argument", pos)
            return None

        if name == "Chars":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, STRING_T):
                self.error("Chars requires string argument", pos)
            return ListT(kind="list", element=STRING_T)

        if name == "ListFrom":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None:
                tu = _unwrap_nil_union(t)
                if isinstance(tu, ListT):
                    return tu
                if isinstance(tu, SetT):
                    return ListT(kind="list", element=tu.element)
                if tu.kind != TY_ERROR:
                    self.error("ListFrom requires list or set argument", pos)
            return None

        # ── Repeat ──
        if name == "Repeat":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t2 is not None:
                t2_rp = _unwrap_nil_union(t2)
            else:
                t2_rp = None
            if t2_rp is not None and not type_eq(t2_rp, INT_T):
                self.error("Repeat count must be int", pos)
            if t1 is not None:
                t1_rp = _unwrap_nil_union(t1)
                if type_eq(t1_rp, STRING_T):
                    return STRING_T
                if type_eq(t1_rp, BYTES_T):
                    return BYTES_T
                if isinstance(t1_rp, ListT):
                    return t1_rp
                if isinstance(t1_rp, TupleT) and t1_rp.elements:
                    return ListT(kind="list", element=t1_rp.elements[0])
                self.error("Repeat requires string, bytes, or list", pos)
            return None

        if name == "Reverse":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, STRING_T):
                self.error("Reverse requires string", pos)
            return STRING_T

        # ── Reversed / Sorted ──
        if name == "Reversed":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and isinstance(t, ListT):
                return t
            if t is not None:
                self.error("Reversed requires list", pos)
            return None
        if name == "Sorted":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None:
                t_so = _unwrap_nil_union(t)
            else:
                t_so = None
            if t_so is not None and isinstance(t_so, ListT):
                if t_so.element.kind not in (
                    TY_INT,
                    TY_FLOAT,
                    TY_BYTE,
                    TY_RUNE,
                    TY_STRING,
                    TY_ERROR,
                ):
                    self.error("Sorted requires ordered type", pos)
                return t_so
            if t_so is not None and isinstance(t_so, SetT):
                if t_so.element.kind not in (
                    TY_INT,
                    TY_FLOAT,
                    TY_BYTE,
                    TY_RUNE,
                    TY_STRING,
                    TY_ERROR,
                ):
                    self.error("Sorted requires ordered type", pos)
                return ListT(kind="list", element=t_so.element)
            if t_so is not None:
                self.error("Sorted requires list or set", pos)
            return None

        # ── String functions ──
        if name in ("Upper", "Lower"):
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None:
                if type_eq(t, BYTES_T):
                    return BYTES_T
                if not type_eq(t, STRING_T):
                    self.error(name + " requires string or bytes", pos)
            return STRING_T
        if name in ("Trim", "TrimStart", "TrimEnd"):
            if not _bctx_require(ctx, 2):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and type_eq(t, BYTES_T):
                return BYTES_T
            return STRING_T
        if name in ("Split", "SplitWhitespace"):
            if name == "Split":
                if not _bctx_require(ctx, 2):
                    return None
                t = _bctx_arg(ctx, 0)
                if (
                    t is not None
                    and t.kind != TY_ERROR
                    and not type_eq(t, STRING_T)
                    and not type_eq(t, BYTES_T)
                ):
                    self.error("Split requires string or bytes as first argument", pos)
                if t is not None and type_eq(t, BYTES_T):
                    return ListT(kind="list", element=BYTES_T)
            else:
                if not _bctx_require(ctx, 1):
                    return None
                t = _bctx_arg(ctx, 0)
                if t is not None and type_eq(t, BYTES_T):
                    return ListT(kind="list", element=BYTES_T)
            return ListT(kind="list", element=STRING_T)
        if name == "SplitN":
            if not _bctx_require(ctx, 3):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and type_eq(t, BYTES_T):
                return ListT(kind="list", element=BYTES_T)
            return ListT(kind="list", element=STRING_T)
        if name == "Join":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            if (
                t1 is not None
                and t1.kind != TY_ERROR
                and not type_eq(t1, STRING_T)
                and not type_eq(t1, BYTES_T)
            ):
                self.error("Join requires string or bytes as first argument", pos)
            t2 = _bctx_arg(ctx, 1)
            if t2 is not None and t2.kind != TY_ERROR:
                if not isinstance(t2, ListT):
                    self.error("Join requires list as second argument", pos)
            if t1 is not None and type_eq(t1, BYTES_T):
                return BYTES_T
            return STRING_T
        if name in ("Find", "RFind"):
            if not _bctx_require(ctx, 2):
                return None
            t = _bctx_arg(ctx, 0)
            if (
                t is not None
                and t.kind != TY_ERROR
                and not type_eq(t, STRING_T)
                and not type_eq(t, BYTES_T)
            ):
                self.error(name + " requires string or bytes as first argument", pos)
            return INT_T
        if name == "Count":
            if not _bctx_require(ctx, 2):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and t.kind != TY_ERROR:
                if (
                    not type_eq(t, STRING_T)
                    and not isinstance(t, ListT)
                    and not type_eq(t, BYTES_T)
                ):
                    self.error(
                        "Count requires string, list, or bytes as first argument",
                        pos,
                    )
            return INT_T
        if name == "Replace":
            if not _bctx_require(ctx, 3):
                return None
            t = _bctx_arg(ctx, 0)
            if (
                t is not None
                and t.kind != TY_ERROR
                and not type_eq(t, STRING_T)
                and not type_eq(t, BYTES_T)
            ):
                self.error("Replace requires string or bytes as first argument", pos)
            if t is not None and type_eq(t, BYTES_T):
                return BYTES_T
            return STRING_T
        if name == "ReplaceCount":
            if not _bctx_require(ctx, 4):
                return None
            t = _bctx_arg(ctx, 0)
            if (
                t is not None
                and t.kind != TY_ERROR
                and not type_eq(t, STRING_T)
                and not type_eq(t, BYTES_T)
            ):
                self.error(
                    "ReplaceCount requires string or bytes as first argument", pos
                )
            if t is not None and type_eq(t, BYTES_T):
                return BYTES_T
            return STRING_T
        if name in ("StartsWith", "EndsWith"):
            if not _bctx_require(ctx, 2):
                return None
            t = _bctx_arg(ctx, 0)
            if (
                t is not None
                and t.kind != TY_ERROR
                and not type_eq(t, STRING_T)
                and not type_eq(t, BYTES_T)
            ):
                self.error(name + " requires string or bytes as first argument", pos)
            return BOOL_T
        if name in ("IsDigit", "IsAlpha", "IsAlnum", "IsSpace", "IsUpper", "IsLower"):
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not (type_eq(t, STRING_T) or type_eq(t, RUNE_T)):
                self.error(name + " requires string or rune", pos)
            return BOOL_T

        # ── RuneFromInt / RuneToInt ──
        if name == "RuneFromInt":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None:
                t_rfi = _unwrap_nil_union(t)
            else:
                t_rfi = None
            if (
                t_rfi is not None
                and t_rfi.kind != TY_ERROR
                and not type_eq(t_rfi, INT_T)
            ):
                self.error("RuneFromInt requires int", pos)
            return RUNE_T
        if name == "RuneToInt":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, RUNE_T):
                self.error("RuneToInt requires rune", pos)
            return INT_T

        # ── ParseInt / ParseFloat ──
        if name == "ParseInt":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None:
                t1_pi = _unwrap_nil_union(t1)
            else:
                t1_pi = None
            if (
                t1_pi is not None
                and t1_pi.kind != TY_ERROR
                and not type_eq(t1_pi, STRING_T)
            ):
                self.error("ParseInt requires string as first argument", pos)
            if t2 is not None:
                t2_pi = _unwrap_nil_union(t2)
            else:
                t2_pi = None
            if (
                t2_pi is not None
                and t2_pi.kind != TY_ERROR
                and not type_eq(t2_pi, INT_T)
            ):
                self.error(
                    "cannot pass " + type_name(t2_pi) + " as int",
                    pos,
                )
            return INT_T
        if name == "ParseFloat":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and t.kind != TY_ERROR and not type_eq(t, STRING_T):
                self.error("ParseFloat requires string", pos)
            return FLOAT_T
        if name == "FormatInt":
            if not _bctx_require(ctx, 2):
                return None
            t1 = _bctx_arg(ctx, 0)
            t2 = _bctx_arg(ctx, 1)
            if t1 is not None:
                t1_fi = _unwrap_nil_union(t1)
            else:
                t1_fi = None
            if (
                t1_fi is not None
                and t1_fi.kind != TY_ERROR
                and not type_eq(t1_fi, INT_T)
            ):
                self.error("FormatInt requires int as first argument", pos)
            if t2 is not None:
                t2_fi = _unwrap_nil_union(t2)
            else:
                t2_fi = None
            if (
                t2_fi is not None
                and t2_fi.kind != TY_ERROR
                and not type_eq(t2_fi, INT_T)
            ):
                self.error("FormatInt requires int as second argument", pos)
            return STRING_T

        # ── Conversions ──
        if name == "IntToFloat":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, INT_T):
                self.error("IntToFloat requires int", pos)
            return FLOAT_T
        if name == "FloatToInt":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, FLOAT_T):
                self.error("FloatToInt requires float", pos)
            return INT_T
        if name == "ByteToInt":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, BYTE_T):
                self.error("ByteToInt requires byte", pos)
            return INT_T
        if name == "IntToByte":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, INT_T):
                self.error("IntToByte requires int", pos)
            return BYTE_T
        if name == "ToString":
            if not _bctx_require(ctx, 1):
                return None
            return STRING_T

        # ── Format ──
        if name == "Format":
            if n < 1:
                self.error("Format requires at least 1 argument", pos)
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and t.kind != TY_ERROR and not type_eq(t, STRING_T):
                self.error("Format template must be string", pos)
            # Check remaining args are all string
            i = 1
            while i < n:
                at = _bctx_arg(ctx, i)
                if at is not None:
                    at_uw = _unwrap_nil_union(at)
                else:
                    at_uw = None
                if (
                    at_uw is not None
                    and at_uw.kind != TY_ERROR
                    and not type_eq(at_uw, STRING_T)
                ):
                    self.error("Format arguments must be string", args[i].pos)
                i += 1
            # Check placeholder count matches arg count
            fmt_val: TExpr = args[0].value
            if isinstance(fmt_val, TStringLit):
                placeholders = fmt_val.value.count("{}")
                arg_count = n - 1
                if placeholders != arg_count:
                    self.error(
                        "Format placeholder count mismatch: "
                        + str(placeholders)
                        + " placeholders, "
                        + str(arg_count)
                        + " arguments",
                        pos,
                    )
            return STRING_T

        # ── I/O ──
        if name in ("WriteOut", "WriteErr", "WritelnOut", "WritelnErr"):
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not (
                type_eq(t, STRING_T) or type_eq(t, BYTES_T) or type_eq(t, ERROR_T)
            ):
                self.error(name + " requires string or bytes", pos)
            return VOID_T
        if name == "ReadLine":
            if not _bctx_require(ctx, 0):
                return None
            return make_optional(STRING_T)
        if name == "ReadAll":
            if not _bctx_require(ctx, 0):
                return None
            return STRING_T
        if name == "ReadBytes":
            if not _bctx_require(ctx, 0):
                return None
            return BYTES_T
        if name == "ReadBytesN":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, INT_T):
                self.error("ReadBytesN requires int", pos)
            return BYTES_T
        if name == "ReadFile":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, STRING_T):
                self.error("ReadFile requires string path", pos)
            return STRING_T
        if name == "ReadFileBytes":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, STRING_T):
                self.error("ReadFileBytes requires string path", pos)
            return BYTES_T
        if name == "WriteFile":
            if not _bctx_require(ctx, 2):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, STRING_T):
                self.error("WriteFile requires string path", pos)
            return VOID_T
        if name == "Args":
            if not _bctx_require(ctx, 0):
                return None
            return ListT(kind="list", element=STRING_T)
        if name == "GetEnv":
            if not _bctx_require(ctx, 1):
                return None
            return make_optional(STRING_T)
        if name == "Exit":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, INT_T):
                self.error("Exit requires int", pos)
            return VOID_T

        # ── Assert ──
        if name == "Assert":
            if not _bctx_require_range(ctx, 1, 2):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, BOOL_T):
                self.error("Assert condition must be bool", pos)
            if n == 2:
                t2 = _bctx_arg(ctx, 1)
                if t2 is not None and not type_eq(t2, STRING_T):
                    self.error("Assert message must be string", pos)
            return VOID_T

        # ── Unwrap ──
        if name == "Unwrap":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None:
                if contains_nil(t):
                    return remove_nil(t)
                self.error("Unwrap requires optional type", pos)
                return t
            return None

        # ── Math extras ──
        if name == "IsNaN" or name == "IsInf":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, FLOAT_T):
                self.error(name + " requires float", pos)
            return BOOL_T
        if name == "Sqrt":
            if not _bctx_require(ctx, 1):
                return None
            t = _bctx_arg(ctx, 0)
            if t is not None and not type_eq(t, FLOAT_T):
                self.error("Sqrt requires float", pos)
            return FLOAT_T

        # ── IsNil ──
        if name == "IsNil":
            if not _bctx_require_range(ctx, 1, 2):
                return None
            return BOOL_T

        # ── IsType ──
        if name == "IsType":
            if not _bctx_require(ctx, 2):
                return None
            _bctx_arg(ctx, 0)
            return BOOL_T

        self.error("unknown built-in function: " + name, pos)
        return None


# ============================================================
# PUBLIC API
# ============================================================


def check(module: TModule) -> list[CheckError]:
    """Type-check a parsed TModule. Returns a list of errors (empty = ok)."""
    checker = Checker()
    checker.strict_math = module.strict_math
    checker.collect_declarations(module)
    if len(checker.errors) > 0:
        return checker.errors
    checker.enter_scope()
    for decl in module.decls:
        if isinstance(decl, TLetStmt):
            checker.check_let_stmt(decl)
    checker.check_bodies(module)
    _check_main(checker)
    return checker.errors


def check_with_info(module: TModule) -> tuple[list[CheckError], Checker]:
    """Type-check and return both errors and the Checker (for downstream passes)."""
    checker = Checker()
    checker.strict_math = module.strict_math
    checker.collect_declarations(module)
    if len(checker.errors) > 0:
        return (checker.errors, checker)
    checker.enter_scope()
    for decl in module.decls:
        if isinstance(decl, TLetStmt):
            checker.check_let_stmt(decl)
    checker.check_bodies(module)
    _check_main(checker)
    return (checker.errors, checker)


def _check_main(checker: Checker) -> None:
    if "Main" not in checker.functions:
        checker.error("missing Main", Pos(1, 1))
        return
    main = checker.functions["Main"]
    if len(main.params) > 0:
        checker.error("Main must take no parameters", Pos(1, 1))
    if not type_eq(main.ret, VOID_T):
        checker.error("Main must return void", Pos(1, 1))
