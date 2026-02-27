"""Generator — orchestrates type-correct Taytsh program generation."""

from __future__ import annotations

from random import Random

from src.taytsh.ast import TModule
from src.taytsh.check import FnT, Type

from .builtins import BuiltinGen
from .decls import DeclGen
from .exprs import ExprGen
from .features import FeatureVector
from .names import NameGen
from .scope import ScopeTracker
from .stmts import StmtGen
from .types import TypePool


class Generator:
    def __init__(self, features: FeatureVector, seed: int) -> None:
        self.features = features
        self.seed = seed
        self.rng = Random(seed)
        self.names = NameGen(self.rng)
        self.pool = TypePool(self.rng, features, self.names)
        self.scope = ScopeTracker()
        self.expr_gen = ExprGen(self)
        self.stmt_gen = StmtGen(self)
        self.builtin_gen = BuiltinGen(self)
        self.decl_gen = DeclGen(self)
        self.functions: dict[str, FnT] = {}
        self.fn_param_names: dict[str, list[str]] = {}
        self.current_fn_ret: Type | None = None
        self.in_loop: bool = False
        self.in_fn_lit: bool = False
        self.in_finally: bool = False

    def generate(self) -> TModule:
        self.pool.build()
        type_decls = self.decl_gen.emit_type_decls()
        fn_decls = self.decl_gen.emit_functions()
        main = self.decl_gen.emit_main()
        all_decls = list(type_decls) + list(fn_decls) + [main]
        return TModule(decls=all_decls)


def generate(features: FeatureVector, seed: int) -> TModule:
    return Generator(features, seed).generate()
