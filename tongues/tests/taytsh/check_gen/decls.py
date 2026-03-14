"""DeclGen — declaration generation (structs, interfaces, enums, functions)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.taytsh.ast import (
    TEnumDecl,
    TFieldDecl,
    TFnDecl,
    TInterfaceDecl,
    TParam,
    TPrimitive,
    TStructDecl,
    TModuleItem,
)
from src.taytsh.check import (
    VOID_T,
    FnT,
    StructT,
    type_eq,
)

from .types import make_ttype
from .ast_helpers import P, A

if TYPE_CHECKING:
    from . import Generator


class DeclGen:
    def __init__(self, gen: Generator) -> None:
        self.gen = gen
        self.rng = gen.rng

    def emit_type_decls(self) -> list[TModuleItem]:
        decls: list[TModuleItem] = []

        # Emit interfaces first (since structs reference them as parents)
        for info in self.gen.pool.interfaces:
            decls.append(
                TInterfaceDecl(
                    pos=P,
                    annotations=A,
                    name=info.resolved.name,
                    fields=[],
                )
            )

        # Emit structs
        for si in self.gen.pool.structs:
            st = si.resolved
            fields: list[TFieldDecl] = []
            for fname, ftype in st.fields.items():
                fields.append(TFieldDecl(pos=P, name=fname, typ=make_ttype(ftype)))

            methods: list[TFnDecl] = []
            if self.gen.features.struct_method and self.rng.random() < 0.5:
                methods = self._gen_struct_methods(st)

            decls.append(
                TStructDecl(
                    pos=P,
                    annotations=A,
                    name=st.name,
                    parent=si.parent,
                    fields=fields,
                    methods=methods,
                )
            )

        # Emit enums
        for ei in self.gen.pool.enums:
            et = ei.resolved
            decls.append(
                TEnumDecl(
                    pos=P, annotations=A, name=et.name, variants=list(et.variants)
                )
            )

        return decls

    def _gen_struct_methods(self, st: StructT) -> list[TFnDecl]:
        methods: list[TFnDecl] = []
        n_methods = self.rng.randint(0, 2)
        for _ in range(n_methods):
            method_name = self.gen.names.method_name()
            ret_type = self.gen.pool.random_value_type()
            params = [TParam(pos=P, name="this", typ=None, annotations=A)]

            n_extra = self.rng.randint(0, 2)
            param_types = []
            used_names: set[str] = {"this"}
            for _ in range(n_extra):
                pt = self.gen.pool.random_value_type()
                pname = self.gen.names.var_name(used_names)
                used_names.add(pname)
                params.append(
                    TParam(pos=P, name=pname, typ=make_ttype(pt), annotations=A)
                )
                param_types.append(pt)

            # Generate body
            self.gen.scope.enter_scope()
            self.gen.scope.declare("this", st, mutable=False)
            for pname, pt in zip([p.name for p in params[1:]], param_types):
                self.gen.scope.declare(pname, pt)

            old_ret = self.gen.current_fn_ret
            self.gen.current_fn_ret = ret_type

            if type_eq(ret_type, VOID_T):
                body = self.gen.stmt_gen.gen_block(
                    self.rng.randint(1, 3), must_return=None
                )
            else:
                body = self.gen.stmt_gen.gen_block(
                    self.rng.randint(1, 3), must_return=ret_type
                )

            self.gen.current_fn_ret = old_ret
            self.gen.scope.exit_scope()

            # Register method in the resolved struct
            fn_type = FnT(kind="fn", params=[st] + param_types, ret=ret_type)
            st.methods[method_name] = fn_type

            methods.append(
                TFnDecl(
                    pos=P,
                    annotations=A,
                    name=method_name,
                    params=params,
                    ret=make_ttype(ret_type),
                    body=body,
                )
            )
        return methods

    def emit_functions(self) -> list[TFnDecl]:
        decls: list[TFnDecl] = []
        n_fns = self.rng.randint(1, 3)
        for _ in range(n_fns):
            fn_decl = self._gen_function()
            if fn_decl is not None:
                decls.append(fn_decl)
        return decls

    def _gen_function(self) -> TFnDecl | None:
        fn_name = self.gen.names.fn_name()
        ret_type = self.gen.pool.random_type(exclude_void=False)
        n_params = self.rng.randint(0, 4)

        params: list[TParam] = []
        param_types: list[tuple[str, type]] = []
        used_names: set[str] = set()
        resolved_params: list[type] = []

        for _ in range(n_params):
            pt = self.gen.pool.random_value_type()
            pname = self.gen.names.var_name(used_names)
            used_names.add(pname)
            params.append(TParam(pos=P, name=pname, typ=make_ttype(pt), annotations=A))
            resolved_params.append(pt)

        # Register function before generating body (for recursive calls)
        fn_t = FnT(kind="fn", params=resolved_params, ret=ret_type)
        self.gen.functions[fn_name] = fn_t
        self.gen.fn_param_names[fn_name] = [p.name for p in params]

        # Generate body
        self.gen.scope.enter_scope()
        for p, pt in zip(params, resolved_params):
            self.gen.scope.declare(p.name, pt)

        old_ret = self.gen.current_fn_ret
        self.gen.current_fn_ret = ret_type

        if type_eq(ret_type, VOID_T):
            body = self.gen.stmt_gen.gen_block(self.rng.randint(2, 5), must_return=None)
        else:
            body = self.gen.stmt_gen.gen_block(
                self.rng.randint(2, 5), must_return=ret_type
            )

        self.gen.current_fn_ret = old_ret
        self.gen.scope.exit_scope()

        return TFnDecl(
            pos=P,
            annotations=A,
            name=fn_name,
            params=params,
            ret=make_ttype(ret_type),
            body=body,
        )

    def emit_main(self) -> TFnDecl:
        self.gen.scope.enter_scope()
        old_ret = self.gen.current_fn_ret
        self.gen.current_fn_ret = VOID_T

        n_stmts = self.rng.randint(3, 8)
        body = self.gen.stmt_gen.gen_block(n_stmts, must_return=None)

        self.gen.current_fn_ret = old_ret
        self.gen.scope.exit_scope()

        return TFnDecl(
            pos=P,
            annotations=A,
            name="Main",
            params=[],
            ret=TPrimitive(pos=P, kind="void"),
            body=body,
        )
