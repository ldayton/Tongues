"""Struct equality tests."""

import sys
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


def test_struct_eq_same() -> None:
    """Two structs with same fields compare equal."""
    a: Point = Point(1, 2)
    b: Point = Point(1, 2)
    assert a == b


def test_struct_eq_different() -> None:
    """Different field values compare not-equal."""
    a: Point = Point(1, 2)
    b: Point = Point(1, 3)
    assert not (a == b)


def test_struct_ne() -> None:
    """!= operator on structs."""
    a: Point = Point(1, 2)
    b: Point = Point(3, 4)
    assert a != b
    assert not (Point(5, 6) != Point(5, 6))


def test_struct_eq_in_list() -> None:
    """Struct equality works with 'in' operator."""
    pts: list[Point] = [Point(1, 2), Point(3, 4)]
    assert Point(1, 2) in pts
    assert Point(3, 4) in pts
    assert Point(5, 6) not in pts


class Item:
    def __init__(self, name: str, value: int) -> None:
        self.name: str = name
        self.value: int = value

    def copy(self) -> "Item":
        return Item(self.name, self.value)

    def doubled(self) -> "Item":
        return Item(self.name, self.value * 2)


class Container:
    def __init__(self) -> None:
        self.items: list[Item] = []

    def add(self, item: Item) -> None:
        self.items.append(item)

    def copy_all(self) -> list[Item]:
        result: list[Item] = []
        for item in self.items:
            result.append(item.copy())
        return result

    def double_all(self) -> list[Item]:
        result: list[Item] = []
        for item in self.items:
            result.append(item.doubled())
        return result


def test_struct_method_on_field_loop_var() -> None:
    """Calling a struct method on a loop variable from iterating a field."""
    c = Container()
    c.add(Item("a", 1))
    c.add(Item("b", 2))
    copies: list[Item] = c.copy_all()
    assert len(copies) == 2
    assert copies[0].name == "a"
    assert copies[0].value == 1
    # Copies must be real Item objects — call a method on them
    doubled: Item = copies[0].doubled()
    assert doubled.value == 2
    # Verify they're independent copies
    copies[0].value = 99
    assert c.items[0].value == 1


def test_struct_noncopying_method_on_field_loop_var() -> None:
    """Calling a non-copy struct method on a loop variable from iterating a field."""
    c = Container()
    c.add(Item("x", 3))
    c.add(Item("y", 5))
    doubled: list[Item] = c.double_all()
    assert len(doubled) == 2
    assert doubled[0].value == 6
    assert doubled[1].value == 10


class Registry:
    """Struct with methods whose names collide with built-in dict/list methods."""

    def __init__(self) -> None:
        self.entries: list[str] = []

    def get(self, idx: int) -> str:
        return self.entries[idx]

    def index(self, name: str) -> int:
        for i in range(len(self.entries)):
            if self.entries[i] == name:
                return i
        return -1

    def keys(self) -> list[str]:
        result: list[str] = []
        for e in self.entries:
            result.append(e)
        return result

    def clear(self) -> None:
        self.entries = []

    def copy(self) -> "Registry":
        r = Registry()
        for e in self.entries:
            r.entries.append(e)
        return r


class Wrapper:
    def __init__(self, reg: Registry) -> None:
        self.reg: Registry = reg


def test_struct_method_name_collides_with_builtin() -> None:
    """Struct methods named get/index/keys/clear/copy dispatch correctly."""
    r = Registry()
    r.entries.append("alpha")
    r.entries.append("beta")
    r.entries.append("gamma")
    assert r.get(0) == "alpha"
    assert r.get(2) == "gamma"
    assert r.index("beta") == 1
    assert r.index("missing") == -1
    k: list[str] = r.keys()
    assert len(k) == 3
    assert k[0] == "alpha"
    c: Registry = r.copy()
    assert c.get(0) == "alpha"
    assert len(c.entries) == 3
    # copy is independent
    c.clear()
    assert len(c.entries) == 0
    assert len(r.entries) == 3


def test_struct_method_via_field_access() -> None:
    """Calling a colliding-name struct method through a field, not a local var."""
    r = Registry()
    r.entries.append("one")
    r.entries.append("two")
    w = Wrapper(r)
    assert w.reg.get(0) == "one"
    assert w.reg.index("two") == 1
    k: list[str] = w.reg.keys()
    assert len(k) == 2
    c: Registry = w.reg.copy()
    c.entries.append("three")
    assert len(c.entries) == 3
    assert len(w.reg.entries) == 2


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_struct_eq_same()
        passed += 1
        print("  PASS test_struct_eq_same")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_struct_eq_same: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_struct_eq_same: " + str(e))
    try:
        test_struct_eq_different()
        passed += 1
        print("  PASS test_struct_eq_different")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_struct_eq_different: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_struct_eq_different: " + str(e))
    try:
        test_struct_ne()
        passed += 1
        print("  PASS test_struct_ne")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_struct_ne: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_struct_ne: " + str(e))
    try:
        test_struct_eq_in_list()
        passed += 1
        print("  PASS test_struct_eq_in_list")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_struct_eq_in_list: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_struct_eq_in_list: " + str(e))
    try:
        test_struct_method_on_field_loop_var()
        passed += 1
        print("  PASS test_struct_method_on_field_loop_var")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_struct_method_on_field_loop_var: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_struct_method_on_field_loop_var: " + str(e))
    try:
        test_struct_noncopying_method_on_field_loop_var()
        passed += 1
        print("  PASS test_struct_noncopying_method_on_field_loop_var")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_struct_noncopying_method_on_field_loop_var: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_struct_noncopying_method_on_field_loop_var: " + str(e))
    try:
        test_struct_method_name_collides_with_builtin()
        passed += 1
        print("  PASS test_struct_method_name_collides_with_builtin")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_struct_method_name_collides_with_builtin: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_struct_method_name_collides_with_builtin: " + str(e))
    try:
        test_struct_method_via_field_access()
        passed += 1
        print("  PASS test_struct_method_via_field_access")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_struct_method_via_field_access: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_struct_method_via_field_access: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
