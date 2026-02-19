"""ScopeTracker — mirrors the checker's scoping rules for the generator."""

from __future__ import annotations

from dataclasses import dataclass

from src.taytsh.check import RESERVED_NAMES, Type, is_assignable


@dataclass
class Binding:
    name: str
    typ: Type
    mutable: bool
    is_loop_var: bool = False


class ScopeTracker:
    def __init__(self) -> None:
        self.scopes: list[dict[str, Binding]] = []

    def enter_scope(self) -> None:
        self.scopes.append({})

    def exit_scope(self) -> None:
        self.scopes.pop()

    def declare(
        self, name: str, typ: Type, *, mutable: bool = True, is_loop_var: bool = False
    ) -> None:
        if len(self.scopes) > 0:
            self.scopes[-1][name] = Binding(name, typ, mutable, is_loop_var)

    def lookup(self, name: str) -> Binding | None:
        i = len(self.scopes) - 1
        while i >= 0:
            if name in self.scopes[i]:
                return self.scopes[i][name]
            i -= 1
        return None

    def narrow(self, name: str, typ: Type) -> None:
        if len(self.scopes) > 0:
            existing = self.lookup(name)
            mutable = existing.mutable if existing is not None else True
            is_loop_var = existing.is_loop_var if existing is not None else False
            self.scopes[-1][name] = Binding(name, typ, mutable, is_loop_var)

    def can_declare(self, name: str) -> bool:
        if name == "_":
            return False
        if name in RESERVED_NAMES:
            return False
        for scope in self.scopes:
            if name in scope:
                return False
        return True

    def bindings_of_type(self, target: Type) -> list[Binding]:
        seen: set[str] = set()
        result: list[Binding] = []
        i = len(self.scopes) - 1
        while i >= 0:
            for name, binding in self.scopes[i].items():
                if name not in seen and is_assignable(binding.typ, target):
                    result.append(binding)
                seen.add(name)
            i -= 1
        return result

    def mutable_bindings(self) -> list[Binding]:
        seen: set[str] = set()
        result: list[Binding] = []
        i = len(self.scopes) - 1
        while i >= 0:
            for name, binding in self.scopes[i].items():
                if name not in seen and binding.mutable and not binding.is_loop_var:
                    result.append(binding)
                seen.add(name)
            i -= 1
        return result

    def all_bindings(self) -> list[Binding]:
        seen: set[str] = set()
        result: list[Binding] = []
        i = len(self.scopes) - 1
        while i >= 0:
            for name, binding in self.scopes[i].items():
                if name not in seen:
                    result.append(binding)
                seen.add(name)
            i -= 1
        return result

    def depth(self) -> int:
        return len(self.scopes)
