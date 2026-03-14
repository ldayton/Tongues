"""Mutator — 22 mutation operators for type checker testing."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from random import Random

from src.taytsh.ast import (
    TArg,
    TAssignStmt,
    TBoolLit,
    TBreakStmt,
    TCall,
    TExpr,
    TExprStmt,
    TFnDecl,
    TFnLit,
    TForStmt,
    TIntLit,
    TLetStmt,
    TMatchStmt,
    TModule,
    TNilLit,
    TOptionalType,
    TPrimitive,
    TReturnStmt,
    TSlice,
    TStmt,
    TStringLit,
    TStructDecl,
    TThrowStmt,
    TTryStmt,
    TVar,
    TWhileStmt,
)
from src.taytsh.check import (
    BUILTIN_NAMES,
)
from .ast_helpers import P, A

_PRIM_KINDS = ["int", "float", "bool", "byte", "bytes", "string", "rune"]


_INCOMPATIBLE_MAP = {
    "int": "string",
    "float": "bool",
    "bool": "bytes",
    "byte": "string",
    "bytes": "int",
    "string": "int",
    "rune": "bytes",
}


def _different_prim(kind: str) -> str:
    return _INCOMPATIBLE_MAP.get(kind, "string")


def _wrong_type_literal(kind: str) -> TExpr:
    if kind == "int":
        return TStringLit(pos=P, annotations=A, value="wrong")
    return TIntLit(pos=P, annotations=A, value=42, raw="42")


def _literal_for(kind: str) -> TExpr:
    if kind == "string":
        return TStringLit(pos=P, annotations=A, value="test")
    if kind == "bool":
        return TBoolLit(pos=P, annotations=A, value=True)
    return TIntLit(pos=P, annotations=A, value=0, raw="0")


# ── AST Walkers ──


def _walk_stmts(stmts: list[TStmt]) -> list[TStmt]:
    """Yield all statements recursively."""
    result: list[TStmt] = []
    for s in stmts:
        result.append(s)
        if isinstance(s, TLetStmt):
            pass
        elif hasattr(s, "then_body"):
            result.extend(_walk_stmts(s.then_body))
            if hasattr(s, "else_body") and s.else_body is not None:
                result.extend(_walk_stmts(s.else_body))
        elif hasattr(s, "body") and isinstance(getattr(s, "body"), list):
            result.extend(_walk_stmts(s.body))
        if isinstance(s, TMatchStmt):
            for case in s.cases:
                result.extend(_walk_stmts(case.body))
            if s.default is not None:
                result.extend(_walk_stmts(s.default.body))
    return result


def _find_fn_decls(module: TModule) -> list[TFnDecl]:
    result: list[TFnDecl] = []
    for d in module.decls:
        if isinstance(d, TFnDecl):
            result.append(d)
        if isinstance(d, TStructDecl):
            result.extend(d.methods)
    return result


def _find_lets(module: TModule) -> list[TLetStmt]:
    result: list[TLetStmt] = []
    for fn in _find_fn_decls(module):
        for s in _walk_stmts(fn.body):
            if isinstance(s, TLetStmt):
                result.append(s)
    return result


def _find_calls(module: TModule) -> list[TCall]:
    result: list[TCall] = []
    for fn in _find_fn_decls(module):
        for s in _walk_stmts(fn.body):
            if isinstance(s, TExprStmt) and isinstance(s.expr, TCall):
                result.append(s.expr)
            if (
                isinstance(s, TLetStmt)
                and s.value is not None
                and isinstance(s.value, TCall)
            ):
                result.append(s.value)
    return result


def _find_returns(module: TModule) -> list[TReturnStmt]:
    result: list[TReturnStmt] = []
    for fn in _find_fn_decls(module):
        for s in _walk_stmts(fn.body):
            if isinstance(s, TReturnStmt) and s.value is not None:
                result.append(s)
    return result


def _find_matches(module: TModule) -> list[TMatchStmt]:
    result: list[TMatchStmt] = []
    for fn in _find_fn_decls(module):
        for s in _walk_stmts(fn.body):
            if isinstance(s, TMatchStmt):
                result.append(s)
    return result


def _find_fn_lits(module: TModule) -> list[TFnLit]:
    result: list[TFnLit] = []
    for fn in _find_fn_decls(module):
        for s in _walk_stmts(fn.body):
            if (
                isinstance(s, TLetStmt)
                and s.value is not None
                and isinstance(s.value, TFnLit)
            ):
                result.append(s.value)
    return result


# ── Mutation Operators ──


@dataclass
class MutationResult:
    name: str
    expected_error: str
    module: TModule


def _try_mutate(
    module: TModule, name: str, expected: str, apply_fn
) -> MutationResult | None:
    m = copy.deepcopy(module)
    if apply_fn(m):
        return MutationResult(name, expected, m)
    return None


def swap_type(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        lets = _find_lets(m)
        targets = [
            l
            for l in lets
            if isinstance(l.typ, TPrimitive)
            and l.typ.kind in _PRIM_KINDS
            and l.value is not None
        ]
        if not targets:
            return False
        let = rng.choice(targets)
        old_kind = let.typ.kind
        new_kind = _different_prim(old_kind)
        let.typ = TPrimitive(pos=P, kind=new_kind)
        return True

    return _try_mutate(module, "swap_type", "cannot assign", apply)


def wrong_arg_type(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        calls = [
            c
            for c in _find_calls(m)
            if not (isinstance(c.func, TVar) and c.func.name in BUILTIN_NAMES)
        ]
        targets = [c for c in calls if len(c.args) > 0]
        if not targets:
            return False
        call = rng.choice(targets)
        idx = rng.randrange(len(call.args))
        # Replace with nil — incompatible with non-optional types
        call.args[idx] = TArg(
            pos=P, name=call.args[idx].name, value=TNilLit(pos=P, annotations=A)
        )
        return True

    return _try_mutate(module, "wrong_arg_type", "cannot", apply)


def wrong_arg_count(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        calls = [
            c
            for c in _find_calls(m)
            if not (isinstance(c.func, TVar) and c.func.name in BUILTIN_NAMES)
        ]
        if not calls:
            return False
        call = rng.choice(calls)
        # Add an extra argument
        call.args.append(
            TArg(
                pos=P,
                name=None,
                value=TIntLit(pos=P, annotations=A, value=99, raw="99"),
            )
        )
        return True

    return _try_mutate(module, "wrong_arg_count", "got", apply)


def wrong_return_type(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        # Find returns paired with their enclosing function's return type
        for fn in _find_fn_decls(m):
            ret_kind = fn.ret.kind if isinstance(fn.ret, TPrimitive) else None
            for s in _walk_stmts(fn.body):
                if isinstance(s, TReturnStmt) and s.value is not None:
                    # Pick a type guaranteed incompatible with the return type
                    if ret_kind == "string":
                        s.value = TIntLit(pos=P, annotations=A, value=42, raw="42")
                    else:
                        s.value = TStringLit(pos=P, annotations=A, value="WRONG_RETURN")
                    return True
        return False

    return _try_mutate(module, "wrong_return_type", "cannot return", apply)


def missing_match_case(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        matches = _find_matches(m)
        targets = [mt for mt in matches if len(mt.cases) > 1 and mt.default is None]
        if not targets:
            return False
        match = rng.choice(targets)
        match.cases.pop()
        return True

    return _try_mutate(module, "missing_match_case", "non-exhaustive match", apply)


def duplicate_match_case(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        matches = _find_matches(m)
        targets = [mt for mt in matches if len(mt.cases) > 0]
        if not targets:
            return False
        match = rng.choice(targets)
        dup = copy.deepcopy(match.cases[0])
        match.cases.append(dup)
        return True

    return _try_mutate(module, "duplicate_match_case", "duplicate case", apply)


def capture_variable(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        fn_lits = _find_fn_lits(m)
        if not fn_lits:
            return False
        # Find a let in the enclosing function and reference it from the fn lit
        for fn in _find_fn_decls(m):
            lets_in_fn = [s for s in fn.body if isinstance(s, TLetStmt)]
            fn_lit_stmts = [
                s
                for s in fn.body
                if isinstance(s, TLetStmt)
                and s.value is not None
                and isinstance(s.value, TFnLit)
            ]
            if lets_in_fn and fn_lit_stmts:
                outer_let = lets_in_fn[0]
                fn_lit_stmt = fn_lit_stmts[0]
                fn_lit = fn_lit_stmt.value
                assert isinstance(fn_lit, TFnLit)
                capture_ref = TExprStmt(
                    pos=P,
                    annotations=A,
                    expr=TVar(pos=P, annotations=A, name=outer_let.name),
                )
                fn_lit.body.insert(0, capture_ref)
                return True
        return False

    return _try_mutate(module, "capture_variable", "cannot capture", apply)


def shadow_binding(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        fns = _find_fn_decls(m)
        for fn in fns:
            lets = [s for s in fn.body if isinstance(s, TLetStmt)]
            if len(lets) >= 2:
                # Shadow the first let using a later position (after second let)
                existing = lets[0]
                second = lets[1]
                dup = TLetStmt(
                    pos=P,
                    annotations=A,
                    name=existing.name,
                    typ=TPrimitive(pos=P, kind="int"),
                    value=TIntLit(pos=P, annotations=A, value=0, raw="0"),
                )
                idx = fn.body.index(second)
                fn.body.insert(idx + 1, dup)
                return True
        return False

    return _try_mutate(module, "shadow_binding", "shadows outer binding", apply)


def use_reserved_name(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        lets = _find_lets(m)
        if not lets:
            return False
        let = rng.choice(lets)
        let.name = "Len"
        return True

    return _try_mutate(module, "use_reserved_name", "reserved name", apply)


def assign_to_this(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        for d in m.decls:
            if isinstance(d, TStructDecl) and d.methods:
                method = d.methods[0]
                stmt = TAssignStmt(
                    pos=P,
                    annotations=A,
                    target=TVar(pos=P, annotations=A, name="this"),
                    value=TIntLit(pos=P, annotations=A, value=0, raw="0"),
                )
                method.body.insert(0, stmt)
                return True
        return False

    return _try_mutate(module, "assign_to_this", "cannot assign to this", apply)


def void_as_value(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        lets = _find_lets(m)
        if not lets:
            return False
        let = rng.choice(lets)
        let.typ = TPrimitive(pos=P, kind="void")
        let.value = None
        return True

    return _try_mutate(module, "void_as_value", "void is not a value type", apply)


def break_outside_loop(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        fns = [d for d in m.decls if isinstance(d, TFnDecl)]
        if not fns:
            return False
        fn = fns[-1]  # Main is usually last
        fn.body.insert(0, TBreakStmt(pos=P, annotations=A))
        return True

    return _try_mutate(module, "break_outside_loop", "break outside of loop", apply)


def use_before_assign(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        for d in m.decls:
            if isinstance(d, TStructDecl):
                fns = [dd for dd in m.decls if isinstance(dd, TFnDecl)]
                if fns:
                    fn = fns[-1]
                    from src.taytsh.ast import TIdentType

                    new_let = TLetStmt(
                        pos=P,
                        annotations=A,
                        name="uninit_var",
                        typ=TIdentType(pos=P, name=d.name),
                        value=None,
                    )
                    # Assign the uninitialized variable to another variable to trigger read
                    read_let = TLetStmt(
                        pos=P,
                        annotations=A,
                        name="uninit_read",
                        typ=TIdentType(pos=P, name=d.name),
                        value=TVar(pos=P, annotations=A, name="uninit_var"),
                    )
                    fn.body.insert(0, new_let)
                    fn.body.insert(1, read_let)
                    return True
        return False

    return _try_mutate(module, "use_before_assign", "used before assignment", apply)


def double_optional(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        lets = _find_lets(m)
        if not lets:
            return False
        let = rng.choice(lets)
        let.typ = TOptionalType(
            pos=P, inner=TOptionalType(pos=P, inner=TPrimitive(pos=P, kind="int"))
        )
        let.value = None
        return True

    return _try_mutate(module, "double_optional", "double optional", apply)


def call_non_function(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        fns = [d for d in m.decls if isinstance(d, TFnDecl)]
        if not fns:
            return False
        fn = fns[-1]
        # Find a let with a primitive type (not struct/fn/interface/enum)
        lets = [
            s
            for s in fn.body
            if isinstance(s, TLetStmt)
            and isinstance(s.typ, TPrimitive)
            and s.typ.kind in _PRIM_KINDS
        ]
        if not lets:
            return False
        let = lets[0]
        call_stmt = TExprStmt(
            pos=P,
            annotations=A,
            expr=TCall(
                pos=P,
                annotations=A,
                func=TVar(pos=P, annotations=A, name=let.name),
                args=[],
            ),
        )
        fn.body.insert(fn.body.index(let) + 1, call_stmt)
        return True

    return _try_mutate(module, "call_non_function", "cannot call", apply)


def mixed_args(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        calls = _find_calls(m)
        targets = [
            c
            for c in calls
            if len(c.args) >= 2 and all(a.name is not None for a in c.args)
        ]
        if not targets:
            return False
        call = rng.choice(targets)
        # Make first arg positional (name=None) and keep rest named -> mixed
        call.args[0] = TArg(pos=P, name=None, value=call.args[0].value)
        return True

    return _try_mutate(module, "mixed_args", "cannot mix positional and named", apply)


def wrong_named_arg(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        calls = _find_calls(m)
        targets = [
            c
            for c in calls
            if len(c.args) > 0 and any(a.name is not None for a in c.args)
        ]
        if not targets:
            return False
        call = rng.choice(targets)
        for i, arg in enumerate(call.args):
            if arg.name is not None:
                call.args[i] = TArg(pos=P, name="nonexistent_param", value=arg.value)
                return True
        return False

    return _try_mutate(module, "wrong_named_arg", "nonexistent_param", apply)


def throw_non_struct(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        fns = _find_fn_decls(m)
        if not fns:
            return False
        fn = fns[-1]
        fn.body.insert(
            0,
            TThrowStmt(
                pos=P,
                annotations=A,
                expr=TIntLit(pos=P, annotations=A, value=42, raw="42"),
            ),
        )
        return True

    return _try_mutate(module, "throw_non_struct", "cannot throw", apply)


def expr_no_effect(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        fns = _find_fn_decls(m)
        if not fns:
            return False
        fn = fns[-1]
        stmt = TExprStmt(
            pos=P,
            annotations=A,
            expr=TIntLit(pos=P, annotations=A, value=42, raw="42"),
        )
        fn.body.insert(0, stmt)
        return True

    return _try_mutate(module, "expr_no_effect", "expression has no effect", apply)


def control_flow_in_finally(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        for fn in _find_fn_decls(m):
            for s in _walk_stmts(fn.body):
                if isinstance(s, TTryStmt) and s.finally_body is not None:
                    s.finally_body.insert(
                        0, TReturnStmt(pos=P, annotations=A, value=None)
                    )
                    return True
        return False

    return _try_mutate(
        module, "control_flow_in_finally", "control flow in finally", apply
    )


def unreachable_code(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        for fn in _find_fn_decls(m):
            for s in _walk_stmts(fn.body):
                if isinstance(s, (TWhileStmt, TForStmt)) and len(s.body) >= 1:
                    for i, stmt in enumerate(s.body):
                        if isinstance(stmt, (TReturnStmt, TBreakStmt)):
                            dead = TLetStmt(
                                pos=P,
                                annotations=A,
                                name="dead_code",
                                typ=TPrimitive(pos=P, kind="int"),
                                value=TIntLit(pos=P, annotations=A, value=0, raw="0"),
                            )
                            s.body.insert(i + 1, dead)
                            return True
        return False

    return _try_mutate(module, "unreachable_code", "unreachable code", apply)


def assign_to_slice(module: TModule, rng: Random) -> MutationResult | None:
    def apply(m: TModule) -> bool:
        for fn in _find_fn_decls(m):
            lets = [s for s in fn.body if isinstance(s, TLetStmt)]
            for let in lets:
                if isinstance(let.typ, TPrimitive) and let.typ.kind == "string":
                    idx = fn.body.index(let)
                    assign = TAssignStmt(
                        pos=P,
                        annotations=A,
                        target=TSlice(
                            pos=P,
                            annotations=A,
                            obj=TVar(pos=P, annotations=A, name=let.name),
                            low=TIntLit(pos=P, annotations=A, value=0, raw="0"),
                            high=TIntLit(pos=P, annotations=A, value=1, raw="1"),
                        ),
                        value=TStringLit(pos=P, annotations=A, value="x"),
                    )
                    fn.body.insert(idx + 1, assign)
                    return True
        return False

    return _try_mutate(module, "assign_to_slice", "cannot assign to slice", apply)


# ── All mutation operators ──


ALL_MUTATIONS = [
    swap_type,
    wrong_arg_type,
    wrong_arg_count,
    wrong_return_type,
    missing_match_case,
    duplicate_match_case,
    capture_variable,
    shadow_binding,
    use_reserved_name,
    assign_to_this,
    void_as_value,
    break_outside_loop,
    use_before_assign,
    call_non_function,
    mixed_args,
    wrong_named_arg,
    throw_non_struct,
    expr_no_effect,
    control_flow_in_finally,
    unreachable_code,
    assign_to_slice,
]
