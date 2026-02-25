"""NameGen — fresh name generation avoiding reserved names and shadowing."""

from __future__ import annotations

from random import Random

from src.taytsh.check import RESERVED_NAMES


_VAR_PREFIXES: list[str] = ["x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h"]
_STRUCT_PREFIXES: list[str] = ["Foo", "Bar", "Baz", "Qux", "Wob", "Zim", "Dex"]
_IFACE_PREFIXES: list[str] = ["Shape", "Node", "Item", "Expr", "Base"]
_ENUM_PREFIXES: list[str] = ["Color", "Dir", "Mode", "Kind", "Flag"]
_FN_PREFIXES: list[str] = ["compute", "process", "handle", "transform", "make"]


class NameGen:
    def __init__(self, rng: Random) -> None:
        self.rng = rng
        self._counters: dict[str, int] = {}

    def _next(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0)
        self._counters[prefix] = n + 1
        if n == 0:
            return prefix
        return prefix + str(n)

    def var_name(self, used: set[str] | None = None) -> str:
        prefix = self.rng.choice(_VAR_PREFIXES)
        name = self._next(prefix)
        while name in RESERVED_NAMES or (used is not None and name in used):
            name = self._next(prefix)
        return name

    def struct_name(self) -> str:
        prefix = self.rng.choice(_STRUCT_PREFIXES)
        name = self._next(prefix)
        while name in RESERVED_NAMES:
            name = self._next(prefix)
        return name

    def interface_name(self) -> str:
        prefix = self.rng.choice(_IFACE_PREFIXES)
        name = self._next(prefix)
        while name in RESERVED_NAMES:
            name = self._next(prefix)
        return name

    def enum_name(self) -> str:
        prefix = self.rng.choice(_ENUM_PREFIXES)
        name = self._next(prefix)
        while name in RESERVED_NAMES:
            name = self._next(prefix)
        return name

    def fn_name(self) -> str:
        prefix = self.rng.choice(_FN_PREFIXES)
        name = self._next(prefix)
        while name in RESERVED_NAMES:
            name = self._next(prefix)
        return name

    def variant_name(self, index: int) -> str:
        names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
        if index < len(names):
            return names[index]
        return "V" + str(index)

    def method_name(self) -> str:
        prefix = self.rng.choice(["doWork", "calc", "run", "get", "step"])
        return self._next(prefix)
