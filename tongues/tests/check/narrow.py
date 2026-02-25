"""Narrowing coverage — taxonomy-driven testing of type narrowing.

Approach: define a cross-product of narrowing dimensions (guard form, control
structure, indirection, type combo), enumerate all valid combinations as specs,
and for each spec generate a pair of small programs:

  accept — uses the narrowed variable at the correct type.  Zero checker
           errors expected.  A failure here is a false negative: the checker
           rejects code that should be valid.

  reject — uses the narrowed variable at the *wrong* type (e.g. accessing a
           field that only exists on the other union member).  At least one
           checker error expected.  A failure here is a false positive: the
           checker accepts code that should be invalid.

This is the "success/failure pair" methodology from the If-T type-narrowing
benchmark [1].  The key insight is that each pair isolates exactly one
narrowing capability: the success case exercises it, the failure case probes
the opposite.  Unlike the random-program generator in tests/check/ (which
builds well-typed programs and mutates them), this module tests whether the
checker can *derive* narrowing from code patterns — the perception problem
rather than the judgment problem.

The spec space is enumerated exhaustively rather than sampled.  Each program
is 5-15 statements, so the full suite runs in <1s.

Dimensions:
  guard       nil_neq, nil_eq, istype, not_istype, isnil, not_isnil
  structure   if_then, if_else, guard_return, while_body, ternary,
              and_chain, or_chain, assert, match, bool_var,
              reassign_then_guard, field_path, nested_narrow,
              double_guard, elif_narrow
  filler      0 or 2 intervening statements between guard and use
  type combo  int?, string?, Shape->Circle, Shape->Square, Shape?->Circle

References:
  [1] Greenman, Dimoulas, Felleisen. "If-T: A Benchmark for Type Narrowing."
      Programming Journal, 2025. https://programming-journal.org/2025/10/17/
  [2] Frank, Quiring, Lampropoulos. "Generating Well-Typed Terms that are
      Not Useless." POPL 2024. (Motivates testing inference, not just
      compatibility.)
  [3] Astral py-fuzzer for ruff/ty: pysource-codegen + crash oracle on
      random valid programs. https://github.com/astral-sh/ruff/blob/main/python/py-fuzzer/fuzz.py
"""

from __future__ import annotations

from dataclasses import dataclass

from src.taytsh.ast import (
    Ann,
    Pos,
    TArg,
    TBinaryOp,
    TBoolLit,
    TCall,
    TExprStmt,
    TFieldAccess,
    TFieldDecl,
    TFnDecl,
    TIfStmt,
    TIntLit,
    TInterfaceDecl,
    TLetStmt,
    TMatchCase,
    TMatchStmt,
    TModule,
    TModuleItem,
    TNilLit,
    TPatternNil,
    TPatternType,
    TPrimitive,
    TReturnStmt,
    TStringLit,
    TStructDecl,
    TTernary,
    TUnaryOp,
    TVar,
    TWhileStmt,
)
from src.taytsh.check import (
    BOOL_T,
    INT_T,
    NIL_T,
    STRING_T,
    InterfaceT,
    StructT,
    Type,
    UnionT,
    check,
    make_optional,
    remove_nil,
    type_eq,
)


from tests.check.types import make_ttype

P = Pos(1, 1)
A: Ann = {}


# ── Type scaffolding ────────────────────────────────────────────

# Shared struct/interface declarations used across all specs.
# Two structs implementing an interface, so we can test IsType narrowing.

_CIRCLE = StructT(
    kind="struct",
    name="Circle",
    fields={"radius": INT_T},
    methods={},
    parent="Shape",
    field_order=["radius"],
)
_SQUARE = StructT(
    kind="struct",
    name="Square",
    fields={"side": INT_T},
    methods={},
    parent="Shape",
    field_order=["side"],
)
_SHAPE = InterfaceT(kind="interface", name="Shape", variants=["Circle", "Square"])

# A struct with a unique field for reject-case testing.
_POINT = StructT(
    kind="struct",
    name="Point",
    fields={"x": INT_T, "y": INT_T},
    methods={},
    parent=None,
    field_order=["x", "y"],
)

_TYPE_DECLS: list[TModuleItem] = [
    TInterfaceDecl(pos=P, name="Shape", annotations=A, fields=[]),
    TStructDecl(
        pos=P,
        name="Circle",
        parent="Shape",
        fields=[TFieldDecl(pos=P, name="radius", typ=TPrimitive(pos=P, kind="int"))],
        methods=[],
        annotations=A,
    ),
    TStructDecl(
        pos=P,
        name="Square",
        parent="Shape",
        fields=[TFieldDecl(pos=P, name="side", typ=TPrimitive(pos=P, kind="int"))],
        methods=[],
        annotations=A,
    ),
    TStructDecl(
        pos=P,
        name="Point",
        parent=None,
        fields=[
            TFieldDecl(pos=P, name="x", typ=TPrimitive(pos=P, kind="int")),
            TFieldDecl(pos=P, name="y", typ=TPrimitive(pos=P, kind="int")),
        ],
        methods=[],
        annotations=A,
    ),
]


# ── Spec definition ─────────────────────────────────────────────


@dataclass
class NarrowingSpec:
    """One narrowing scenario to test."""

    name: str
    # What type the variable starts as
    input_type: Type
    # What type the variable narrows to in the positive branch
    narrowed_type: Type
    # What guard form to use
    guard: str  # "nil_neq", "nil_eq", "istype", "not_istype", "assert_nil", "assert_istype", "isnil", "not_isnil"
    # What control structure delivers the narrowing
    structure: str  # "if_then", "if_else", "guard_return", "while_body", "ternary", "and_chain", "or_chain", "match"
    # How many filler statements between guard and use (0 = immediate)
    filler: int


@dataclass
class NarrowingFailure:
    """A test failure."""

    spec_name: str
    case: str  # "accept" or "reject"
    expected_clean: bool
    actual_errors: list[str]


# ── AST helpers ──────────────────────────────────────────────────


def _var(name: str) -> TVar:
    return TVar(pos=P, name=name, annotations=A)


def _int(n: int) -> TIntLit:
    return TIntLit(pos=P, value=n, raw=str(n), annotations=A)


def _nil() -> TNilLit:
    return TNilLit(pos=P, annotations=A)


def _bool(v: bool) -> TBoolLit:
    return TBoolLit(pos=P, value=v, annotations=A)


def _str(s: str) -> TStringLit:
    return TStringLit(pos=P, value=s, annotations=A)


def _let(name: str, typ: Type, value=None) -> TLetStmt:
    return TLetStmt(pos=P, name=name, typ=make_ttype(typ), value=value, annotations=A)


def _return(value=None) -> TReturnStmt:
    return TReturnStmt(pos=P, value=value, annotations=A)


def _expr_stmt(expr) -> TExprStmt:
    return TExprStmt(pos=P, expr=expr, annotations=A)


def _call(func_name: str, *args) -> TCall:
    targs = [TArg(pos=P, name=None, value=a) for a in args]
    return TCall(pos=P, func=_var(func_name), args=targs, annotations=A)


def _field(obj, field: str) -> TFieldAccess:
    return TFieldAccess(pos=P, obj=obj, field=field, annotations=A)


def _if(cond, then_body, else_body=None) -> TIfStmt:
    return TIfStmt(
        pos=P, cond=cond, then_body=then_body, else_body=else_body, annotations=A
    )


def _while(cond, body) -> TWhileStmt:
    return TWhileStmt(pos=P, cond=cond, body=body, annotations=A)


def _not(expr) -> TUnaryOp:
    return TUnaryOp(pos=P, op="!", operand=expr, annotations=A)


def _binop(left, op, right) -> TBinaryOp:
    return TBinaryOp(pos=P, op=op, left=left, right=right, annotations=A)


def _ternary(cond, then_expr, else_expr) -> TTernary:
    return TTernary(
        pos=P, cond=cond, then_expr=then_expr, else_expr=else_expr, annotations=A
    )


def _filler_stmts(n: int) -> list:
    """Generate n harmless filler statements to increase distance between guard and use."""
    stmts = []
    for i in range(n):
        stmts.append(_let(f"_filler{i}", INT_T, _int(i)))
    return stmts


# ── Condition builders ───────────────────────────────────────────


def _nil_neq_cond(var_name: str) -> TBinaryOp:
    """x != nil"""
    return _binop(_var(var_name), "!=", _nil())


def _nil_eq_cond(var_name: str) -> TBinaryOp:
    """x == nil"""
    return _binop(_var(var_name), "==", _nil())


def _istype_cond(var_name: str, type_name: str) -> TCall:
    """IsType(x, "T")"""
    return _call("IsType", _var(var_name), _str(type_name))


def _not_istype_cond(var_name: str, type_name: str) -> TUnaryOp:
    """!IsType(x, "T")"""
    return _not(_istype_cond(var_name, type_name))


def _isnil_cond(var_name: str) -> TCall:
    """IsNil(x)"""
    return _call("IsNil", _var(var_name))


def _not_isnil_cond(var_name: str) -> TUnaryOp:
    """!IsNil(x)"""
    return _not(_isnil_cond(var_name))


# ── Use-site builders ───────────────────────────────────────────


def _use_at_type(var_name: str, typ: Type) -> list:
    """Generate a statement that uses var_name assuming it has the given type.

    Returns a list of statements. The usage must be something the checker
    validates — field access on a struct, arithmetic on int, etc.
    """
    if isinstance(typ, StructT):
        # Access the first field
        field_name = next(iter(typ.fields))
        return [
            _let("_used", typ.fields[field_name], _field(_var(var_name), field_name))
        ]
    if type_eq(typ, INT_T):
        return [_let("_used", INT_T, _binop(_var(var_name), "+", _int(1)))]
    if type_eq(typ, STRING_T):
        return [_let("_used", INT_T, _call("Len", _var(var_name)))]
    if isinstance(typ, InterfaceT):
        # Match on the interface — this requires it to be an interface, not a union
        return [_let("_used", INT_T, _int(0))]
    # Fallback: just reference it
    return [_let("_used", typ, _var(var_name))]


def _use_at_type_named(var_name: str, typ: Type, bind_name: str) -> list:
    """Like _use_at_type but with a custom binding name to avoid shadowing."""
    if isinstance(typ, StructT):
        field_name = next(iter(typ.fields))
        return [
            _let(bind_name, typ.fields[field_name], _field(_var(var_name), field_name))
        ]
    if type_eq(typ, INT_T):
        return [_let(bind_name, INT_T, _binop(_var(var_name), "+", _int(1)))]
    if type_eq(typ, STRING_T):
        return [_let(bind_name, INT_T, _call("Len", _var(var_name)))]
    return [_let(bind_name, typ, _var(var_name))]


def _use_wrong_type(var_name: str, narrowed_type: Type, input_type: Type) -> list:
    """Generate a statement that is only valid for the NON-narrowed part of input_type.

    This should produce a type error if narrowing works correctly (the variable
    has been narrowed away from the type we're trying to use it as).
    """
    if type_eq(narrowed_type, NIL_T):
        # Variable was narrowed to nil; try to use it as the non-nil part
        inner = remove_nil(input_type)
        if isinstance(inner, StructT):
            field_name = next(iter(inner.fields))
            return [
                _let(
                    "_wrong",
                    inner.fields[field_name],
                    _field(_var(var_name), field_name),
                )
            ]
        if type_eq(inner, INT_T):
            return [_let("_wrong", INT_T, _binop(_var(var_name), "+", _int(1)))]
    # Variable was narrowed to non-nil or a specific struct; try to use nil operations
    # or access a field that doesn't exist on the narrowed type
    if isinstance(narrowed_type, StructT) and "radius" in narrowed_type.fields:
        # Narrowed to Circle — try accessing "side" (Square's field)
        return [_let("_wrong", INT_T, _field(_var(var_name), "side"))]
    if isinstance(narrowed_type, StructT) and "side" in narrowed_type.fields:
        # Narrowed to Square — try accessing "radius" (Circle's field)
        return [_let("_wrong", INT_T, _field(_var(var_name), "radius"))]
    # For non-nil narrowing of optionals: try to pass to a function expecting the full union
    # Simplest: just declare a let with the wrong type
    return [_let("_wrong", NIL_T, _var(var_name))]


# ── Narrowing type name for IsType ───────────────────────────────


def _istype_target(narrowed_type: Type) -> str | None:
    if isinstance(narrowed_type, StructT):
        return narrowed_type.name
    return None


# ── Module builder ───────────────────────────────────────────────


def _build_narrowing_module(
    spec: NarrowingSpec,
    use_stmts: list,
) -> TModule:
    """Build a module that declares a variable of spec.input_type, narrows it
    via the specified guard/structure, then executes use_stmts in the narrowed context.
    """
    var = "val"
    body: list = []

    # Declare the variable (uninitialized — the checker allows this for narrowing tests)
    body.append(_let(var, spec.input_type))

    filler = _filler_stmts(spec.filler)

    if spec.structure == "if_then":
        cond = _make_guard_cond(spec, var)
        if cond is None:
            return _empty_module()
        then_body = filler + use_stmts
        body.append(_if(cond, then_body))

    elif spec.structure == "if_else":
        # The narrowing we want is in the else branch, so we invert the guard
        cond = _make_inverted_guard_cond(spec, var)
        if cond is None:
            return _empty_module()
        then_body = [_let("_skip", INT_T, _int(0))]
        else_body = filler + use_stmts
        body.append(_if(cond, then_body, else_body))

    elif spec.structure == "guard_return":
        # if <inverted_guard> { return }
        # <filler>
        # <use>  ← narrowed in continuation
        inv_cond = _make_inverted_guard_cond(spec, var)
        if inv_cond is None:
            return _empty_module()
        body.append(_if(inv_cond, [_return()]))
        body.extend(filler)
        body.extend(use_stmts)

    elif spec.structure == "while_body":
        cond = _make_guard_cond(spec, var)
        if cond is None:
            return _empty_module()
        # Only nil checks narrow in while conditions
        while_body = (
            filler + use_stmts + [TReturnStmt(pos=P, value=None, annotations=A)]
        )
        body.append(_while(cond, while_body))

    elif spec.structure == "ternary":
        cond = _make_guard_cond(spec, var)
        if cond is None:
            return _empty_module()
        # Build a ternary that uses the narrowed var in the then-expr
        then_expr = _make_narrowed_use_expr(var, spec.narrowed_type)
        else_expr = _int(0)
        body.append(_let("_result", INT_T, _ternary(cond, then_expr, else_expr)))

    elif spec.structure == "and_chain":
        # x != nil && use(x)
        cond = _make_guard_cond(spec, var)
        if cond is None:
            return _empty_module()
        use_expr = _make_narrowed_use_expr(var, spec.narrowed_type)
        # use_expr needs to be bool for && to work
        # So we do: x != nil && Len(x) > 0 or similar
        right = (
            _binop(use_expr, ">", _int(0))
            if not type_eq(spec.narrowed_type, BOOL_T)
            else use_expr
        )
        chain = _binop(cond, "&&", right)
        body.append(_let("_result", BOOL_T, chain))

    elif spec.structure == "or_chain":
        # x == nil || use(x)
        # || narrowing: if left is true we short-circuit, so right sees inverted narrowing
        inv_cond = _make_inverted_guard_cond(spec, var)
        if inv_cond is None:
            return _empty_module()
        use_expr = _make_narrowed_use_expr(var, spec.narrowed_type)
        right = (
            _binop(use_expr, ">", _int(0))
            if not type_eq(spec.narrowed_type, BOOL_T)
            else use_expr
        )
        chain = _binop(inv_cond, "||", right)
        body.append(_let("_result", BOOL_T, chain))

    elif spec.structure == "bool_var":
        # let notNil: bool = x != nil
        # if notNil { use(x) }
        cond = _make_guard_cond(spec, var)
        if cond is None:
            return _empty_module()
        body.append(_let("_cond", BOOL_T, cond))
        body.append(_if(_var("_cond"), filler + use_stmts))

    elif spec.structure == "reassign_then_guard":
        # let tmp: optional = x
        # if tmp != nil { use(tmp) }
        body.append(_let("tmp", spec.input_type, _var(var)))
        cond = _make_guard_cond_for(spec.guard, "tmp", spec.narrowed_type)
        if cond is None:
            return _empty_module()
        # Rewrite use_stmts to use "tmp" instead of "val"
        tmp_use = _use_at_type("tmp", spec.narrowed_type)
        body.append(_if(cond, filler + tmp_use))

    elif spec.structure == "field_path":
        # let obj: Point = ...  (Point has field x: int, but we make a wrapper)
        # Actually: use a struct with an optional field, narrow through field path
        return _build_field_path_module(spec, use_stmts, filler)

    elif spec.structure == "nested_narrow":
        # if x != nil { if IsType(x, "Circle") { use(x.radius) } }
        # Only meaningful when input_type contains nil
        if not _contains_nil(spec.input_type):
            return _empty_module()
        outer_cond = _nil_neq_cond(var)
        inner_cond = _make_guard_cond(spec, var)
        if inner_cond is None:
            return _empty_module()
        inner_if = _if(inner_cond, filler + use_stmts)
        body.append(_if(outer_cond, [inner_if]))

    elif spec.structure == "double_guard":
        # if x == nil { return }
        # if y == nil { return }
        # use(x)  ← narrowed by first guard
        # use(y)  ← narrowed by second guard
        inv_cond = _make_inverted_guard_cond(spec, var)
        if inv_cond is None:
            return _empty_module()
        body.append(_let("val2", spec.input_type))
        inv_cond2 = _make_inverted_guard_cond(spec, "val2")
        body.append(_if(inv_cond, [_return()]))
        body.append(_if(inv_cond2, [_return()]))
        body.extend(filler)
        body.extend(use_stmts)
        # Use val2 with a different binding name to avoid shadowing
        body.extend(_use_at_type_named("val2", spec.narrowed_type, "_used2"))

    elif spec.structure == "elif_narrow":
        # if x == nil { skip } else { use(x) }
        # This is if/else but the inverted cond is implicit (else of nil check)
        inv_cond = _make_inverted_guard_cond(spec, var)
        if inv_cond is None:
            return _empty_module()
        body.append(_if(inv_cond, [_let("_skip", INT_T, _int(0))], filler + use_stmts))

    elif spec.structure == "assert":
        cond = _make_guard_cond(spec, var)
        if cond is None:
            return _empty_module()
        body.append(_expr_stmt(_call("Assert", cond)))
        body.extend(filler)
        body.extend(use_stmts)

    elif spec.structure == "match":
        body = _build_match_body(spec, var, use_stmts, filler)

    else:
        return _empty_module()

    main = TFnDecl(
        pos=P,
        name="Main",
        params=[],
        ret=TPrimitive(pos=P, kind="void"),
        body=body,
        annotations=A,
    )
    return TModule(decls=list(_TYPE_DECLS) + [main])


def _build_match_body(spec, var, use_stmts, filler):
    """Build body for match-based narrowing."""
    body = [_let(var, spec.input_type)]

    if isinstance(spec.input_type, InterfaceT):
        # Match on interface, use narrowed type in case body
        target_name = _istype_target(spec.narrowed_type)
        if target_name is None:
            return body

        cases = []
        for vname in spec.input_type.variants:
            if vname == target_name:
                # This is the case we care about
                pat = TPatternType(
                    pos=P,
                    name="narrowed",
                    type_name=make_ttype(spec.narrowed_type),
                    annotations=A,
                )
                cases.append(
                    TMatchCase(
                        pos=P, pattern=pat, body=filler + use_stmts, annotations=A
                    )
                )
            else:
                # Other cases get empty bodies
                other_st = _CIRCLE if vname == "Circle" else _SQUARE
                pat = TPatternType(
                    pos=P,
                    name="other",
                    type_name=make_ttype(other_st),
                    annotations=A,
                )
                cases.append(TMatchCase(pos=P, pattern=pat, body=[], annotations=A))

        body.append(
            TMatchStmt(
                pos=P,
                expr=_var(var),
                cases=cases,
                default=None,
                annotations=A,
            )
        )

    elif isinstance(spec.input_type, UnionT):
        # Match on union with optional
        nil_in_union = any(type_eq(m, NIL_T) for m in spec.input_type.members)
        cases = []
        for m in spec.input_type.members:
            if type_eq(m, NIL_T):
                pat = TPatternNil(pos=P)
                if type_eq(spec.narrowed_type, NIL_T):
                    cases.append(
                        TMatchCase(
                            pos=P, pattern=pat, body=filler + use_stmts, annotations=A
                        )
                    )
                else:
                    cases.append(TMatchCase(pos=P, pattern=pat, body=[], annotations=A))
            else:
                pat = TPatternType(
                    pos=P,
                    name="narrowed",
                    type_name=make_ttype(m),
                    annotations=A,
                )
                if type_eq(m, spec.narrowed_type):
                    cases.append(
                        TMatchCase(
                            pos=P, pattern=pat, body=filler + use_stmts, annotations=A
                        )
                    )
                else:
                    cases.append(TMatchCase(pos=P, pattern=pat, body=[], annotations=A))

        body.append(
            TMatchStmt(
                pos=P,
                expr=_var(var),
                cases=cases,
                default=None,
                annotations=A,
            )
        )

    return body


_HOLDER = StructT(
    kind="struct",
    name="Holder",
    fields={"data": make_optional(INT_T)},
    methods={},
    parent=None,
    field_order=["data"],
)

_HOLDER_DECL = TStructDecl(
    pos=P,
    name="Holder",
    parent=None,
    fields=[TFieldDecl(pos=P, name="data", typ=make_ttype(make_optional(INT_T)))],
    methods=[],
    annotations=A,
)


def _build_field_path_module(spec, use_stmts, filler):
    """Build module testing field-path narrowing: if obj.field != nil { use(obj.field) }."""
    body: list = []
    body.append(_let("obj", _HOLDER))
    # Condition: obj.data != nil
    cond = _binop(_field(_var("obj"), "data"), "!=", _nil())
    # In narrowed context, obj.data should be int
    narrowed_use = [
        _let("_used", INT_T, _binop(_field(_var("obj"), "data"), "+", _int(1)))
    ]
    body.append(_if(cond, filler + narrowed_use))

    decls = list(_TYPE_DECLS) + [_HOLDER_DECL]
    main = TFnDecl(
        pos=P,
        name="Main",
        params=[],
        ret=TPrimitive(pos=P, kind="void"),
        body=body,
        annotations=A,
    )
    return TModule(decls=decls + [main])


def _empty_module() -> TModule:
    main = TFnDecl(
        pos=P,
        name="Main",
        params=[],
        ret=TPrimitive(pos=P, kind="void"),
        body=[],
        annotations=A,
    )
    return TModule(decls=list(_TYPE_DECLS) + [main])


# ── Guard condition builders ─────────────────────────────────────


def _make_guard_cond(spec: NarrowingSpec, var: str):
    """Build the condition that narrows var to spec.narrowed_type."""
    g = spec.guard
    if g == "nil_neq":
        return _nil_neq_cond(var)
    if g == "nil_eq":
        return _nil_eq_cond(var)
    if g == "istype":
        tn = _istype_target(spec.narrowed_type)
        return _istype_cond(var, tn) if tn else None
    if g == "not_istype":
        # Negative istype: !IsType(x, T) means we DON'T match T
        # For guard_return: if !IsType(x, T) { return } → x is T after
        tn = _istype_target(spec.narrowed_type)
        return _not_istype_cond(var, tn) if tn else None
    if g == "assert_nil":
        return _nil_neq_cond(var)
    if g == "assert_istype":
        tn = _istype_target(spec.narrowed_type)
        return _istype_cond(var, tn) if tn else None
    if g == "isnil":
        return _isnil_cond(var)
    if g == "not_isnil":
        return _not_isnil_cond(var)
    return None


def _make_inverted_guard_cond(spec: NarrowingSpec, var: str):
    """Build the inverted condition (for else-branch / guard-return narrowing)."""
    g = spec.guard
    if g == "nil_neq":
        return _nil_eq_cond(var)
    if g == "nil_eq":
        return _nil_neq_cond(var)
    if g == "istype":
        tn = _istype_target(spec.narrowed_type)
        return _not_istype_cond(var, tn) if tn else None
    if g == "not_istype":
        tn = _istype_target(spec.narrowed_type)
        return _istype_cond(var, tn) if tn else None
    if g == "isnil":
        return _isnil_cond(var)
    if g == "not_isnil":
        return _not_isnil_cond(var)
    return None


def _contains_nil(t: Type) -> bool:
    if type_eq(t, NIL_T):
        return True
    if isinstance(t, UnionT):
        return any(type_eq(m, NIL_T) for m in t.members)
    return False


def _make_guard_cond_for(guard: str, var: str, narrowed_type: Type):
    """Build a guard condition for a specific variable name (not from spec)."""
    if guard == "nil_neq":
        return _nil_neq_cond(var)
    if guard == "istype":
        tn = _istype_target(narrowed_type)
        return _istype_cond(var, tn) if tn else None
    return None


def _make_narrowed_use_expr(var_name: str, narrowed_type: Type):
    """Return an expression that is only valid if var_name has narrowed_type.

    Must return an int-typed expression for use in ternary/&& contexts.
    """
    if isinstance(narrowed_type, StructT):
        field_name = next(iter(narrowed_type.fields))
        return _field(_var(var_name), field_name)
    if type_eq(narrowed_type, INT_T):
        return _binop(_var(var_name), "+", _int(1))
    if type_eq(narrowed_type, STRING_T):
        return _call("Len", _var(var_name))
    return _int(0)


# ── Spec enumeration ─────────────────────────────────────────────

# Guard forms and what input/narrowed types they apply to
_GUARD_TYPE_COMBOS: list[tuple[str, Type, Type]] = [
    # Nil narrowing: optional int → int (non-nil branch)
    ("nil_neq", make_optional(INT_T), INT_T),
    # Nil narrowing: optional string → string
    ("nil_neq", make_optional(STRING_T), STRING_T),
    # Nil narrowing via IsNil: if IsNil(x) { return } → x narrowed to non-nil
    ("isnil", make_optional(INT_T), INT_T),
    # !IsNil in if-then: if !IsNil(x) { use(x as non-nil) }
    ("not_isnil", make_optional(INT_T), INT_T),
    # IsType narrowing: Shape → Circle
    ("istype", _SHAPE, _CIRCLE),
    # IsType narrowing: Shape → Square
    ("istype", _SHAPE, _SQUARE),
    # Optional interface: Shape? → Circle (needs nil check then istype)
    ("istype", make_optional(_SHAPE), _CIRCLE),
]

# Structures that support each guard form
_GUARD_STRUCTURES: dict[str, list[str]] = {
    "nil_neq": [
        "if_then",
        "if_else",
        "guard_return",
        "while_body",
        "ternary",
        "and_chain",
        "or_chain",
        "assert",
        # New patterns that test inference limitations:
        "bool_var",  # let ok = x != nil; if ok { use(x) }
        "reassign_then_guard",  # let tmp = x; if tmp != nil { use(tmp) }
        "field_path",  # if obj.field != nil { use(obj.field) }
        "double_guard",  # if x == nil { return }; if y == nil { return }; use both
        "elif_narrow",  # if x == nil { skip } else { use(x) }
    ],
    "nil_eq": ["if_then"],
    "istype": [
        "if_then",
        "if_else",
        "guard_return",
        "ternary",
        "and_chain",
        "match",
        # New patterns:
        "bool_var",  # let ok = IsType(x, "T"); if ok { use(x.field) }
        "reassign_then_guard",  # let tmp = x; if IsType(tmp, "T") { use(tmp.field) }
        "nested_narrow",  # if x != nil { if IsType(x, "T") { use(x.field) } }
    ],
    "not_istype": ["guard_return"],
    "isnil": ["guard_return"],
    # !IsNil(x) in if-then should narrow x to non-nil (same as x != nil)
    "not_isnil": ["if_then"],
}

_FILLER_COUNTS = [0, 2]


def enumerate_specs() -> list[NarrowingSpec]:
    """Enumerate all valid narrowing specs."""
    specs: list[NarrowingSpec] = []
    for guard, input_type, narrowed_type in _GUARD_TYPE_COMBOS:
        structures = _GUARD_STRUCTURES.get(guard, [])
        for structure in structures:
            for filler in _FILLER_COUNTS:
                # Skip filler > 0 for structures where it doesn't make sense
                if filler > 0 and structure in ("ternary", "and_chain", "or_chain"):
                    continue
                # Skip match for optional interfaces — match on interface
                # narrows to interface, not to a specific struct
                if structure == "match" and isinstance(input_type, UnionT):
                    continue
                name = f"{guard}__{structure}__filler{filler}__{_type_label(input_type)}_to_{_type_label(narrowed_type)}"
                specs.append(
                    NarrowingSpec(
                        name=name,
                        input_type=input_type,
                        narrowed_type=narrowed_type,
                        guard=guard,
                        structure=structure,
                        filler=filler,
                    )
                )
    return specs


def _type_label(t: Type) -> str:
    if isinstance(t, StructT):
        return t.name
    if isinstance(t, InterfaceT):
        return t.name
    if isinstance(t, UnionT):
        parts = []
        for m in t.members:
            parts.append(_type_label(m))
        return "_or_".join(parts)
    return t.kind


# ── Runner ───────────────────────────────────────────────────────


def run_narrowing_spec(spec: NarrowingSpec) -> list[NarrowingFailure]:
    """Run one narrowing spec, returning failures (if any)."""
    failures: list[NarrowingFailure] = []

    # Accept case: use the narrowed type correctly
    accept_use = _use_at_type("val", spec.narrowed_type)
    accept_module = _build_narrowing_module(spec, accept_use)
    accept_errors = check(accept_module)
    accept_msgs = [e.msg for e in accept_errors]
    # Filter out unrelated errors
    accept_msgs = _filter_errors(accept_msgs)
    if accept_msgs:
        failures.append(
            NarrowingFailure(
                spec_name=spec.name,
                case="accept",
                expected_clean=True,
                actual_errors=accept_msgs,
            )
        )

    # Reject case: use the wrong type (only for structures where we can
    # construct a meaningful wrong-type access)
    if spec.structure not in (
        "ternary",
        "and_chain",
        "or_chain",
        "bool_var",
        "reassign_then_guard",
        "field_path",
        "double_guard",
        "nested_narrow",
        "elif_narrow",
    ):
        reject_use = _use_wrong_type("val", spec.narrowed_type, spec.input_type)
        reject_module = _build_narrowing_module(spec, reject_use)
        reject_errors = check(reject_module)
        reject_msgs = [e.msg for e in reject_errors]
        reject_msgs = _filter_errors(reject_msgs)
        # We expect at least one error
        if not reject_msgs:
            failures.append(
                NarrowingFailure(
                    spec_name=spec.name,
                    case="reject",
                    expected_clean=False,
                    actual_errors=[],
                )
            )

    return failures


def _filter_errors(msgs: list[str]) -> list[str]:
    """Remove errors that aren't related to narrowing."""
    return [
        m
        for m in msgs
        if "missing Main" not in m and "variable used before assignment" not in m
    ]


def run_all() -> tuple[list[NarrowingFailure], int]:
    """Run all narrowing specs. Returns (failures, total_specs)."""
    specs = enumerate_specs()
    all_failures: list[NarrowingFailure] = []
    for spec in specs:
        all_failures.extend(run_narrowing_spec(spec))
    return all_failures, len(specs)


# ── Reporting ────────────────────────────────────────────────────


def report() -> None:
    """Run all specs and print a summary."""
    failures, total = run_all()
    accept_failures = [f for f in failures if f.case == "accept"]
    reject_failures = [f for f in failures if f.case == "reject"]

    print(f"Narrowing coverage: {total} specs")
    print(f"  Accept failures (false negatives): {len(accept_failures)}")
    print(f"  Reject failures (false positives): {len(reject_failures)}")

    if accept_failures:
        print("\n── Accept failures (checker rejects valid narrowing) ──")
        for f in accept_failures:
            print(f"  {f.spec_name}")
            for e in f.actual_errors:
                print(f"    {e}")

    if reject_failures:
        print("\n── Reject failures (checker misses broken narrowing) ──")
        for f in reject_failures:
            print(f"  {f.spec_name}")

    if not failures:
        print("\nAll narrowing specs passed!")


ALL_SPECS = enumerate_specs()


if __name__ == "__main__":
    report()
