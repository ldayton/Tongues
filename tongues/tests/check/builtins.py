"""BuiltinGen — generate calls to Taytsh built-in functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.taytsh.ast import (
    Pos,
    TArg,
    TCall,
    TExpr,
    TIntLit,
    TVar,
)
from src.taytsh.check import (
    BOOL_T,
    BYTE_T,
    BYTES_T,
    FLOAT_T,
    INT_T,
    RUNE_T,
    STRING_T,
    ListT,
    MapT,
    SetT,
    TupleT,
    Type,
    type_eq,
)

if TYPE_CHECKING:
    from . import Generator

P = Pos(1, 1)
A: dict[str, str] = {}


class BuiltinGen:
    def __init__(self, gen: Generator) -> None:
        self.gen = gen
        self.rng = gen.rng

    def _bindings(self) -> list:
        return self.gen.expr_gen._accessible_bindings()

    def gen_builtin(self, target: Type, depth: int) -> TExpr | None:
        """Try to generate a builtin call returning `target`. Returns None if none fit."""
        candidates: list[tuple[str, callable]] = []
        if type_eq(target, INT_T):
            candidates = self._int_builtins(depth)
        elif type_eq(target, FLOAT_T):
            candidates = self._float_builtins(depth)
        elif type_eq(target, BOOL_T):
            candidates = self._bool_builtins(depth)
        elif type_eq(target, STRING_T):
            candidates = self._string_builtins(depth)
        elif type_eq(target, BYTES_T):
            candidates = self._bytes_builtins(depth)
        elif type_eq(target, RUNE_T):
            candidates = self._rune_builtins(depth)
        elif isinstance(target, ListT) and type_eq(target.element, STRING_T):
            candidates = self._list_string_builtins(depth)
        elif isinstance(target, TupleT) and len(target.elements) == 2:
            if type_eq(target.elements[0], INT_T) and type_eq(
                target.elements[1], INT_T
            ):
                candidates = self._tuple_int_int_builtins(depth)
        if not candidates:
            return None
        _name, factory = self.rng.choice(candidates)
        return factory()

    def gen_void_builtin(self, depth: int) -> TExpr | None:
        """Generate a void builtin call suitable for expr-stmts."""
        candidates: list[tuple[str, callable]] = []
        bindings = self._bindings()
        for b in bindings:
            if isinstance(b.typ, ListT):
                candidates.append(
                    (
                        "Append",
                        lambda bn=b.name, bt=b.typ: self._call(
                            "Append",
                            [
                                self._var(bn),
                                self.gen.expr_gen.gen_expr(bt.element, depth + 1),
                            ],
                        ),
                    )
                )
            if isinstance(b.typ, MapT):
                candidates.append(
                    (
                        "Delete",
                        lambda bn=b.name, bt=b.typ: self._call(
                            "Delete",
                            [
                                self._var(bn),
                                self.gen.expr_gen.gen_expr(bt.key, depth + 1),
                            ],
                        ),
                    )
                )
            if isinstance(b.typ, SetT):
                candidates.append(
                    (
                        "Add",
                        lambda bn=b.name, bt=b.typ: self._call(
                            "Add",
                            [
                                self._var(bn),
                                self.gen.expr_gen.gen_expr(bt.element, depth + 1),
                            ],
                        ),
                    )
                )
                candidates.append(
                    (
                        "Remove",
                        lambda bn=b.name, bt=b.typ: self._call(
                            "Remove",
                            [
                                self._var(bn),
                                self.gen.expr_gen.gen_expr(bt.element, depth + 1),
                            ],
                        ),
                    )
                )
        candidates.append(
            (
                "WriteOut",
                lambda: self._call(
                    "WriteOut", [self.gen.expr_gen.gen_expr(STRING_T, depth + 1)]
                ),
            )
        )
        candidates.append(
            (
                "WritelnOut",
                lambda: self._call(
                    "WritelnOut", [self.gen.expr_gen.gen_expr(STRING_T, depth + 1)]
                ),
            )
        )
        candidates.append(
            (
                "Assert",
                lambda: self._call(
                    "Assert", [self.gen.expr_gen.gen_expr(BOOL_T, depth + 1)]
                ),
            )
        )
        if not candidates:
            return None
        _name, factory = self.rng.choice(candidates)
        return factory()

    # ── Return type: int ──

    def _int_builtins(self, depth: int) -> list[tuple[str, callable]]:
        result: list[tuple[str, callable]] = []
        e = self.gen.expr_gen
        # Len(coll)
        for b in self._bindings():
            if (
                isinstance(b.typ, (ListT, MapT, SetT))
                or type_eq(b.typ, STRING_T)
                or type_eq(b.typ, BYTES_T)
            ):
                result.append(
                    ("Len", lambda bn=b.name: self._call("Len", [self._var(bn)]))
                )
                break
        # IndexOf(list, v)
        for b in self._bindings():
            if isinstance(b.typ, ListT):
                result.append(
                    (
                        "IndexOf",
                        lambda bn=b.name, bt=b.typ: self._call(
                            "IndexOf",
                            [self._var(bn), e.gen_expr(bt.element, depth + 1)],
                        ),
                    )
                )
                break
        result.append(
            ("Round", lambda: self._call("Round", [e.gen_expr(FLOAT_T, depth + 1)]))
        )
        result.append(
            ("Floor", lambda: self._call("Floor", [e.gen_expr(FLOAT_T, depth + 1)]))
        )
        result.append(
            ("Ceil", lambda: self._call("Ceil", [e.gen_expr(FLOAT_T, depth + 1)]))
        )
        result.append(
            (
                "Find",
                lambda: self._call(
                    "Find",
                    [e.gen_expr(STRING_T, depth + 1), e.gen_expr(STRING_T, depth + 1)],
                ),
            )
        )
        result.append(
            (
                "Count",
                lambda: self._call(
                    "Count",
                    [e.gen_expr(STRING_T, depth + 1), e.gen_expr(STRING_T, depth + 1)],
                ),
            )
        )
        result.append(
            (
                "RuneToInt",
                lambda: self._call("RuneToInt", [e.gen_expr(RUNE_T, depth + 1)]),
            )
        )
        result.append(
            (
                "ByteToInt",
                lambda: self._call("ByteToInt", [e.gen_expr(BYTE_T, depth + 1)]),
            )
        )
        result.append(
            (
                "FloatToInt",
                lambda: self._call("FloatToInt", [e.gen_expr(FLOAT_T, depth + 1)]),
            )
        )
        result.append(
            (
                "ParseInt",
                lambda: self._call(
                    "ParseInt",
                    [
                        e.gen_expr(STRING_T, depth + 1),
                        TIntLit(pos=P, value=10, raw="10", annotations=A),
                    ],
                ),
            )
        )
        return result

    # ── Return type: float ──

    def _float_builtins(self, depth: int) -> list[tuple[str, callable]]:
        e = self.gen.expr_gen
        return [
            (
                "IntToFloat",
                lambda: self._call("IntToFloat", [e.gen_expr(INT_T, depth + 1)]),
            ),
            ("Sqrt", lambda: self._call("Sqrt", [e.gen_expr(FLOAT_T, depth + 1)])),
            ("Abs", lambda: self._call("Abs", [e.gen_expr(FLOAT_T, depth + 1)])),
            (
                "ParseFloat",
                lambda: self._call("ParseFloat", [e.gen_expr(STRING_T, depth + 1)]),
            ),
        ]

    # ── Return type: bool ──

    def _bool_builtins(self, depth: int) -> list[tuple[str, callable]]:
        e = self.gen.expr_gen
        result: list[tuple[str, callable]] = []
        # Contains(coll, v)
        for b in self._bindings():
            if isinstance(b.typ, ListT):
                result.append(
                    (
                        "Contains",
                        lambda bn=b.name, bt=b.typ: self._call(
                            "Contains",
                            [self._var(bn), e.gen_expr(bt.element, depth + 1)],
                        ),
                    )
                )
                break
            if isinstance(b.typ, SetT):
                result.append(
                    (
                        "Contains",
                        lambda bn=b.name, bt=b.typ: self._call(
                            "Contains",
                            [self._var(bn), e.gen_expr(bt.element, depth + 1)],
                        ),
                    )
                )
                break
            if type_eq(b.typ, STRING_T):
                result.append(
                    (
                        "Contains",
                        lambda bn=b.name: self._call(
                            "Contains", [self._var(bn), e.gen_expr(STRING_T, depth + 1)]
                        ),
                    )
                )
                break
        result.append(
            (
                "StartsWith",
                lambda: self._call(
                    "StartsWith",
                    [e.gen_expr(STRING_T, depth + 1), e.gen_expr(STRING_T, depth + 1)],
                ),
            )
        )
        result.append(
            (
                "EndsWith",
                lambda: self._call(
                    "EndsWith",
                    [e.gen_expr(STRING_T, depth + 1), e.gen_expr(STRING_T, depth + 1)],
                ),
            )
        )
        result.append(
            (
                "IsDigit",
                lambda: self._call("IsDigit", [e.gen_expr(STRING_T, depth + 1)]),
            )
        )
        result.append(
            (
                "IsAlpha",
                lambda: self._call("IsAlpha", [e.gen_expr(STRING_T, depth + 1)]),
            )
        )
        result.append(
            ("IsNaN", lambda: self._call("IsNaN", [e.gen_expr(FLOAT_T, depth + 1)]))
        )
        result.append(
            ("IsInf", lambda: self._call("IsInf", [e.gen_expr(FLOAT_T, depth + 1)]))
        )
        return result

    # ── Return type: string ──

    def _string_builtins(self, depth: int) -> list[tuple[str, callable]]:
        e = self.gen.expr_gen
        result: list[tuple[str, callable]] = []
        result.append(
            (
                "Concat",
                lambda: self._call(
                    "Concat",
                    [e.gen_expr(STRING_T, depth + 1), e.gen_expr(STRING_T, depth + 1)],
                ),
            )
        )
        result.append(
            ("Upper", lambda: self._call("Upper", [e.gen_expr(STRING_T, depth + 1)]))
        )
        result.append(
            ("Lower", lambda: self._call("Lower", [e.gen_expr(STRING_T, depth + 1)]))
        )
        result.append(
            ("ToString", lambda: self._call("ToString", [e.gen_expr(INT_T, depth + 1)]))
        )
        result.append(
            ("Decode", lambda: self._call("Decode", [e.gen_expr(BYTES_T, depth + 1)]))
        )
        result.append(
            (
                "Replace",
                lambda: self._call(
                    "Replace",
                    [
                        e.gen_expr(STRING_T, depth + 1),
                        e.gen_expr(STRING_T, depth + 1),
                        e.gen_expr(STRING_T, depth + 1),
                    ],
                ),
            )
        )
        result.append(
            (
                "Repeat",
                lambda: self._call(
                    "Repeat",
                    [e.gen_expr(STRING_T, depth + 1), e.gen_expr(INT_T, depth + 1)],
                ),
            )
        )
        result.append(
            (
                "Reverse",
                lambda: self._call("Reverse", [e.gen_expr(STRING_T, depth + 1)]),
            )
        )
        # Join(sep, list[string]) — need a list[string] binding
        for b in self._bindings():
            if isinstance(b.typ, ListT) and type_eq(b.typ.element, STRING_T):
                result.append(
                    (
                        "Join",
                        lambda bn=b.name: self._call(
                            "Join", [e.gen_expr(STRING_T, depth + 1), self._var(bn)]
                        ),
                    )
                )
                break
        return result

    # ── Return type: bytes ──

    def _bytes_builtins(self, depth: int) -> list[tuple[str, callable]]:
        e = self.gen.expr_gen
        return [
            ("Encode", lambda: self._call("Encode", [e.gen_expr(STRING_T, depth + 1)])),
            ("Bytes", lambda: self._call("Bytes", [e.gen_expr(INT_T, depth + 1)])),
            (
                "Concat",
                lambda: self._call(
                    "Concat",
                    [e.gen_expr(BYTES_T, depth + 1), e.gen_expr(BYTES_T, depth + 1)],
                ),
            ),
        ]

    # ── Return type: rune ──

    def _rune_builtins(self, depth: int) -> list[tuple[str, callable]]:
        e = self.gen.expr_gen
        return [
            (
                "RuneFromInt",
                lambda: self._call("RuneFromInt", [e.gen_expr(INT_T, depth + 1)]),
            ),
        ]

    # ── Return type: list[string] ──

    def _list_string_builtins(self, depth: int) -> list[tuple[str, callable]]:
        e = self.gen.expr_gen
        return [
            (
                "Split",
                lambda: self._call(
                    "Split",
                    [e.gen_expr(STRING_T, depth + 1), e.gen_expr(STRING_T, depth + 1)],
                ),
            ),
            (
                "SplitWhitespace",
                lambda: self._call(
                    "SplitWhitespace", [e.gen_expr(STRING_T, depth + 1)]
                ),
            ),
        ]

    # ── Return type: (int, int) ──

    def _tuple_int_int_builtins(self, depth: int) -> list[tuple[str, callable]]:
        e = self.gen.expr_gen
        return [
            (
                "DivMod",
                lambda: self._call(
                    "DivMod",
                    [e.gen_expr(INT_T, depth + 1), e.gen_expr(INT_T, depth + 1)],
                ),
            ),
        ]

    # ── Helpers ──

    def _call(self, name: str, arg_exprs: list[TExpr]) -> TCall:
        args = [TArg(pos=P, name=None, value=v) for v in arg_exprs]
        return TCall(
            pos=P,
            func=TVar(pos=P, name=name, annotations=A),
            args=args,
            annotations=A,
        )

    def _var(self, name: str) -> TVar:
        return TVar(pos=P, name=name, annotations=A)
