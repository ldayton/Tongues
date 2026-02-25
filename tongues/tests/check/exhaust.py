"""Exhaustiveness enumerator — bounded exhaustive match pattern testing."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

from src.taytsh.ast import (
    Pos,
    TDefault,
    TEnumDecl,
    TFnDecl,
    TInterfaceDecl,
    TLetStmt,
    TMatchCase,
    TMatchStmt,
    TModule,
    TModuleItem,
    TPrimitive,
    TPattern,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TStructDecl,
    TVar,
    TFieldDecl,
)
from src.taytsh.check import (
    INT_T,
    NIL_T,
    STRING_T,
    EnumT,
    InterfaceT,
    StructT,
    Type,
    UnionT,
    type_eq,
)

from .types import make_ttype

P = Pos(1, 1)
A: dict[str, str] = {}


@dataclass
class CaseSpec:
    """A possible case in a match statement."""

    key: str
    pattern: TPattern


@dataclass
class TypeConfig:
    name: str
    decls: list[TModuleItem]
    scrutinee_type: Type
    all_cases: list[CaseSpec]
    is_exhaustive: Callable[[list[CaseSpec]], bool]


def _make_let_and_match(
    scrutinee_type: Type, var_name: str, cases: list[CaseSpec], with_default: bool
) -> list:
    let_stmt = TLetStmt(
        pos=P,
        name=var_name,
        typ=make_ttype(scrutinee_type),
        value=None,
        annotations=A,
    )
    match_cases = [
        TMatchCase(pos=P, pattern=c.pattern, body=[], annotations=A) for c in cases
    ]
    default = (
        TDefault(pos=P, name=None, body=[], annotations=A) if with_default else None
    )
    match_stmt = TMatchStmt(
        pos=P,
        expr=TVar(pos=P, name=var_name, annotations=A),
        cases=match_cases,
        default=default,
        annotations=A,
    )
    return [let_stmt, match_stmt]


def _build_module(
    config: TypeConfig, cases: list[CaseSpec], with_default: bool
) -> TModule:
    body = _make_let_and_match(config.scrutinee_type, "val", cases, with_default)
    main = TFnDecl(
        pos=P,
        name="Main",
        params=[],
        ret=TPrimitive(pos=P, kind="void"),
        body=body,
        annotations=A,
    )
    return TModule(decls=list(config.decls) + [main])


def _powerset(items: list[CaseSpec]) -> list[list[CaseSpec]]:
    result: list[list[CaseSpec]] = []
    n = len(items)
    for r in range(n + 1):
        for combo in combinations(items, r):
            result.append(list(combo))
    return result


# ── Oracle functions (independent of the checker) ──


def _enum_exhaustive(enum: EnumT) -> Callable[[list[CaseSpec]], bool]:
    def check(cases: list[CaseSpec]) -> bool:
        covered = {c.key for c in cases}
        for v in enum.variants:
            if enum.name + "." + v not in covered:
                return False
        return True

    return check


def _interface_exhaustive(iface: InterfaceT) -> Callable[[list[CaseSpec]], bool]:
    def check(cases: list[CaseSpec]) -> bool:
        covered = {c.key for c in cases}
        for v in iface.variants:
            if v not in covered:
                return False
        return True

    return check


def _union_exhaustive(
    union: UnionT, interface_variants: dict[str, list[str]]
) -> Callable[[list[CaseSpec]], bool]:
    def check(cases: list[CaseSpec]) -> bool:
        covered = {c.key for c in cases}
        for m in union.members:
            if type_eq(m, NIL_T):
                if "nil" not in covered:
                    return False
            elif isinstance(m, InterfaceT):
                # Interface member covered if interface name itself covered OR all its variants
                if m.name in covered:
                    continue
                variants = interface_variants.get(m.name, [])
                if not all(v in covered for v in variants):
                    return False
            elif isinstance(m, EnumT):
                all_covered = True
                for v in m.variants:
                    if m.name + "." + v not in covered:
                        all_covered = False
                        break
                if not all_covered:
                    return False
            elif isinstance(m, StructT):
                if m.name not in covered:
                    return False
            else:
                # Primitive member
                key = m.kind
                if key not in covered:
                    return False
        return True

    return check


def _optional_exhaustive(inner_type: Type) -> Callable[[list[CaseSpec]], bool]:
    def check(cases: list[CaseSpec]) -> bool:
        covered = {c.key for c in cases}
        has_nil = "nil" in covered
        # Inner type must be covered
        if isinstance(inner_type, (StructT, InterfaceT)):
            has_inner = inner_type.name in covered
        else:
            has_inner = inner_type.kind in covered
        return has_nil and has_inner

    return check


# ── Predefined configs ──


def _simple_enum_config() -> TypeConfig:
    et = EnumT(kind="enum", name="Color", variants=["Red", "Green", "Blue"])
    decl = TEnumDecl(
        pos=P, name="Color", variants=["Red", "Green", "Blue"], annotations=A
    )
    cases = [
        CaseSpec("Color.Red", TPatternEnum(pos=P, enum_name="Color", variant="Red")),
        CaseSpec(
            "Color.Green", TPatternEnum(pos=P, enum_name="Color", variant="Green")
        ),
        CaseSpec("Color.Blue", TPatternEnum(pos=P, enum_name="Color", variant="Blue")),
    ]
    return TypeConfig("simple_enum", [decl], et, cases, _enum_exhaustive(et))


def _large_enum_config() -> TypeConfig:
    variants = ["A", "B", "C", "D", "E", "F"]
    et = EnumT(kind="enum", name="Big", variants=variants)
    decl = TEnumDecl(pos=P, name="Big", variants=variants, annotations=A)
    cases = [
        CaseSpec("Big." + v, TPatternEnum(pos=P, enum_name="Big", variant=v))
        for v in variants
    ]
    return TypeConfig("large_enum", [decl], et, cases, _enum_exhaustive(et))


def _simple_interface_config() -> TypeConfig:
    s1 = StructT(
        kind="struct",
        name="Circle",
        fields={"r": INT_T},
        methods={},
        parent="Shape",
        field_order=["r"],
    )
    s2 = StructT(
        kind="struct",
        name="Square",
        fields={"s": INT_T},
        methods={},
        parent="Shape",
        field_order=["s"],
    )
    it = InterfaceT(kind="interface", name="Shape", variants=["Circle", "Square"])
    decls: list[TModuleItem] = [
        TInterfaceDecl(pos=P, name="Shape", annotations=A, fields=[]),
        TStructDecl(
            pos=P,
            name="Circle",
            parent="Shape",
            fields=[TFieldDecl(pos=P, name="r", typ=TPrimitive(pos=P, kind="int"))],
            methods=[],
            annotations=A,
        ),
        TStructDecl(
            pos=P,
            name="Square",
            parent="Shape",
            fields=[TFieldDecl(pos=P, name="s", typ=TPrimitive(pos=P, kind="int"))],
            methods=[],
            annotations=A,
        ),
    ]
    cases = [
        CaseSpec(
            "Circle",
            TPatternType(pos=P, name="c", type_name=make_ttype(s1), annotations=A),
        ),
        CaseSpec(
            "Square",
            TPatternType(pos=P, name="s", type_name=make_ttype(s2), annotations=A),
        ),
    ]
    return TypeConfig("simple_interface", decls, it, cases, _interface_exhaustive(it))


def _interface_4_config() -> TypeConfig:
    names = ["Lit", "Bin", "Neg", "Call"]
    structs = []
    decls: list[TModuleItem] = [
        TInterfaceDecl(pos=P, name="Expr", annotations=A, fields=[])
    ]
    for n in names:
        st = StructT(
            kind="struct",
            name=n,
            fields={"v": INT_T},
            methods={},
            parent="Expr",
            field_order=["v"],
        )
        structs.append(st)
        decls.append(
            TStructDecl(
                pos=P,
                name=n,
                parent="Expr",
                fields=[TFieldDecl(pos=P, name="v", typ=TPrimitive(pos=P, kind="int"))],
                methods=[],
                annotations=A,
            )
        )
    it = InterfaceT(kind="interface", name="Expr", variants=names)
    cases = [
        CaseSpec(
            n, TPatternType(pos=P, name="e", type_name=make_ttype(st), annotations=A)
        )
        for n, st in zip(names, structs)
    ]
    return TypeConfig("interface_4", decls, it, cases, _interface_exhaustive(it))


def _simple_union_config() -> TypeConfig:
    union = UnionT(kind="union", members=[INT_T, STRING_T])
    cases = [
        CaseSpec(
            "int",
            TPatternType(
                pos=P, name="n", type_name=TPrimitive(pos=P, kind="int"), annotations=A
            ),
        ),
        CaseSpec(
            "string",
            TPatternType(
                pos=P,
                name="s",
                type_name=TPrimitive(pos=P, kind="string"),
                annotations=A,
            ),
        ),
    ]
    return TypeConfig("simple_union", [], union, cases, _union_exhaustive(union, {}))


def _union_3_config() -> TypeConfig:
    from src.taytsh.check import BOOL_T

    union = UnionT(kind="union", members=[INT_T, STRING_T, BOOL_T])
    cases = [
        CaseSpec(
            "int",
            TPatternType(
                pos=P, name="n", type_name=TPrimitive(pos=P, kind="int"), annotations=A
            ),
        ),
        CaseSpec(
            "string",
            TPatternType(
                pos=P,
                name="s",
                type_name=TPrimitive(pos=P, kind="string"),
                annotations=A,
            ),
        ),
        CaseSpec(
            "bool",
            TPatternType(
                pos=P, name="b", type_name=TPrimitive(pos=P, kind="bool"), annotations=A
            ),
        ),
    ]
    return TypeConfig("union_3", [], union, cases, _union_exhaustive(union, {}))


def _optional_config() -> TypeConfig:
    from src.taytsh.check import make_optional

    opt = make_optional(INT_T)
    cases = [
        CaseSpec(
            "int",
            TPatternType(
                pos=P, name="n", type_name=TPrimitive(pos=P, kind="int"), annotations=A
            ),
        ),
        CaseSpec("nil", TPatternNil(pos=P)),
    ]
    return TypeConfig("optional_int", [], opt, cases, _optional_exhaustive(INT_T))


def _union_with_interface_config() -> TypeConfig:
    s1 = StructT(
        kind="struct",
        name="Lit",
        fields={"v": INT_T},
        methods={},
        parent="Node",
        field_order=["v"],
    )
    s2 = StructT(
        kind="struct",
        name="Bin",
        fields={"v": INT_T},
        methods={},
        parent="Node",
        field_order=["v"],
    )
    it = InterfaceT(kind="interface", name="Node", variants=["Lit", "Bin"])
    union = UnionT(kind="union", members=[it, INT_T])
    decls: list[TModuleItem] = [
        TInterfaceDecl(pos=P, name="Node", annotations=A, fields=[]),
        TStructDecl(
            pos=P,
            name="Lit",
            parent="Node",
            fields=[TFieldDecl(pos=P, name="v", typ=TPrimitive(pos=P, kind="int"))],
            methods=[],
            annotations=A,
        ),
        TStructDecl(
            pos=P,
            name="Bin",
            parent="Node",
            fields=[TFieldDecl(pos=P, name="v", typ=TPrimitive(pos=P, kind="int"))],
            methods=[],
            annotations=A,
        ),
    ]
    # Cases can be: int, Node (interface), Lit (struct), Bin (struct)
    cases = [
        CaseSpec(
            "int",
            TPatternType(
                pos=P, name="n", type_name=TPrimitive(pos=P, kind="int"), annotations=A
            ),
        ),
        CaseSpec(
            "Node",
            TPatternType(pos=P, name="nd", type_name=make_ttype(it), annotations=A),
        ),
        CaseSpec(
            "Lit",
            TPatternType(pos=P, name="lt", type_name=make_ttype(s1), annotations=A),
        ),
        CaseSpec(
            "Bin",
            TPatternType(pos=P, name="bn", type_name=make_ttype(s2), annotations=A),
        ),
    ]
    oracle = _union_exhaustive(union, {"Node": ["Lit", "Bin"]})
    return TypeConfig("union_with_interface", decls, union, cases, oracle)


def _optional_enum_config() -> TypeConfig:
    from src.taytsh.check import make_optional

    et = EnumT(kind="enum", name="Dir", variants=["Up", "Down", "Left", "Right"])
    opt = make_optional(et)
    decl = TEnumDecl(
        pos=P, name="Dir", variants=["Up", "Down", "Left", "Right"], annotations=A
    )
    cases = [
        CaseSpec("Dir.Up", TPatternEnum(pos=P, enum_name="Dir", variant="Up")),
        CaseSpec("Dir.Down", TPatternEnum(pos=P, enum_name="Dir", variant="Down")),
        CaseSpec("Dir.Left", TPatternEnum(pos=P, enum_name="Dir", variant="Left")),
        CaseSpec("Dir.Right", TPatternEnum(pos=P, enum_name="Dir", variant="Right")),
        CaseSpec("nil", TPatternNil(pos=P)),
    ]

    def oracle(selected: list[CaseSpec]) -> bool:
        covered = {c.key for c in selected}
        has_nil = "nil" in covered
        has_all_variants = all("Dir." + v in covered for v in et.variants)
        return has_nil and has_all_variants

    return TypeConfig("optional_enum", [decl], opt, cases, oracle)


ALL_CONFIGS: list[TypeConfig] = [
    _simple_enum_config(),
    _large_enum_config(),
    _simple_interface_config(),
    _interface_4_config(),
    _simple_union_config(),
    _union_3_config(),
    _optional_config(),
    _union_with_interface_config(),
    _optional_enum_config(),
]


@dataclass
class ExhaustivenessFailure:
    config_name: str
    cases: list[str]
    with_default: bool
    expected_accept: bool
    actual_errors: list[str]


def run_exhaustiveness(config: TypeConfig) -> list[ExhaustivenessFailure]:
    from src.taytsh.check import check

    failures: list[ExhaustivenessFailure] = []
    subsets = _powerset(config.all_cases)
    for subset in subsets:
        expected_exhaust = config.is_exhaustive(subset)

        # Without default: accept iff exhaustive
        module = _build_module(config, subset, with_default=False)
        errors = check(module)
        error_msgs = [e.msg for e in errors]
        has_exhaust_error = any("non-exhaustive" in m for m in error_msgs)
        # Filter to only exhaustiveness-relevant errors
        other_errors = [
            m
            for m in error_msgs
            if "non-exhaustive" not in m
            and "missing Main" not in m
            and "variable used before assignment" not in m
        ]

        if expected_exhaust and has_exhaust_error:
            failures.append(
                ExhaustivenessFailure(
                    config.name,
                    [c.key for c in subset],
                    False,
                    True,
                    error_msgs,
                )
            )
        elif not expected_exhaust and not has_exhaust_error and not other_errors:
            failures.append(
                ExhaustivenessFailure(
                    config.name,
                    [c.key for c in subset],
                    False,
                    False,
                    error_msgs,
                )
            )

        # With default: should always accept (non-exhaustiveness covered)
        if not expected_exhaust:
            module_d = _build_module(config, subset, with_default=True)
            errors_d = check(module_d)
            error_msgs_d = [e.msg for e in errors_d]
            exhaust_errors = [m for m in error_msgs_d if "non-exhaustive" in m]
            if exhaust_errors:
                failures.append(
                    ExhaustivenessFailure(
                        config.name,
                        [c.key for c in subset],
                        True,
                        True,
                        error_msgs_d,
                    )
                )

    return failures
