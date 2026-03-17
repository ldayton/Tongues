"""Dict object tests."""

import sys


def test_dict_equality() -> None:
    """Dict equality comparisons."""
    assert {"a": 1, "b": 2} == {"a": 1, "b": 2}
    assert {"a": 1, "b": 2} == {"b": 2, "a": 1}
    assert {} == {}
    assert {"a": 1} == {"a": 1}
    assert not ({"a": 1} == {"a": 2})
    assert {"a": 1} != {"a": 2}
    assert {"a": 1} != {"b": 1}
    assert {"a": 1} != {"a": 1, "b": 2}


def test_dict_length() -> None:
    """len() returns key count."""
    assert len({}) == 0
    assert len({"a": 1}) == 1
    assert len({"a": 1, "b": 2, "c": 3}) == 3


def test_dict_get_item() -> None:
    """Indexing returns value for key."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    assert d["a"] == 1
    assert d["b"] == 2
    assert d["c"] == 3


def test_dict_set_item() -> None:
    """Index assignment sets value."""
    d: dict[str, int] = {"a": 1}
    d["b"] = 2
    assert d["b"] == 2
    d["a"] = 10
    assert d["a"] == 10
    assert d == {"a": 10, "b": 2}


def test_dict_set_item_empty() -> None:
    """Adding to empty dict."""
    d: dict[str, int] = {}
    d["a"] = 1
    assert d == {"a": 1}
    d["b"] = 2
    assert d == {"a": 1, "b": 2}


def test_dict_contains() -> None:
    """Membership testing with in."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    assert "a" in d
    assert "b" in d
    assert "c" in d
    assert "d" not in d
    assert "d" not in d


def test_dict_contains_empty() -> None:
    """Membership in empty dict."""
    d: dict[str, int] = {}
    assert "a" not in d
    assert "a" not in d


def test_dict_bool() -> None:
    """Dict truthiness - empty is falsy."""
    assert bool({"a": 1}) == True
    assert bool({}) == False
    assert not {}
    assert {"x": 0}


def test_dict_get() -> None:
    """get() returns value or default."""
    d: dict[str, int] = {"a": 1, "b": 2}
    assert d.get("a") == 1
    assert d.get("b") == 2
    assert d.get("c") is None
    assert d.get("c", 0) == 0
    assert d.get("c", -1) == -1
    assert d.get("a", 99) == 1


def test_dict_get_none_value() -> None:
    """get() with None as actual value."""
    d: dict[str, int | None] = {"a": None, "b": 1}
    assert d.get("a") is None
    assert d.get("a", 99) is None
    assert d.get("c") is None
    assert d.get("c", 99) == 99


def test_dict_keys() -> None:
    """keys() returns key view."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    keys: list[str] = list(d.keys())
    assert len(keys) == 3
    assert "a" in keys
    assert "b" in keys
    assert "c" in keys


def test_dict_list_from_dict() -> None:
    """list(d) returns list of keys."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    ks: list[str] = list(d)
    assert len(ks) == 3
    assert "a" in ks
    assert "b" in ks
    assert "c" in ks


def test_dict_set_from_dict() -> None:
    """set(d) returns set of keys."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    s: set[str] = set(d)
    assert s == {"a", "b", "c"}


def test_dict_list_from_empty_dict() -> None:
    """list(d) on empty dict returns empty list."""
    d: dict[str, int] = {}
    ks: list[str] = list(d)
    assert ks == []


def test_dict_set_from_empty_dict() -> None:
    """set(d) on empty dict returns empty set."""
    d: dict[str, int] = {}
    s: set[str] = set(d)
    assert s == set()


def test_dict_values() -> None:
    """values() returns value view."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    vals: list[int] = list(d.values())
    assert len(vals) == 3
    assert 1 in vals
    assert 2 in vals
    assert 3 in vals


def test_dict_items() -> None:
    """items() returns key-value pairs."""
    d: dict[str, int] = {"a": 1, "b": 2}
    items: list[tuple[str, int]] = list(d.items())
    assert len(items) == 2
    assert ("a", 1) in items
    assert ("b", 2) in items


def test_dict_pop() -> None:
    """pop() removes and returns value."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    x: int = d.pop("b")
    assert x == 2
    assert d == {"a": 1, "c": 3}
    assert "b" not in d


def test_dict_pop_default() -> None:
    """pop() with default for missing key."""
    d: dict[str, int] = {"a": 1}
    x: int = d.pop("b", 99)
    assert x == 99
    assert d == {"a": 1}


def test_dict_setdefault() -> None:
    """setdefault() gets or sets value."""
    d: dict[str, int] = {"a": 1}
    x: int = d.setdefault("a", 99)
    assert x == 1
    assert d["a"] == 1
    x = d.setdefault("b", 99)
    assert x == 99
    assert d["b"] == 99
    assert d == {"a": 1, "b": 99}


def test_dict_update() -> None:
    """update() merges dicts."""
    d: dict[str, int] = {"a": 1, "b": 2}
    d.update({"b": 20, "c": 3})
    assert d == {"a": 1, "b": 20, "c": 3}
    d.update({})
    assert d == {"a": 1, "b": 20, "c": 3}


def test_dict_update_empty() -> None:
    """update() on empty dict."""
    d: dict[str, int] = {}
    d.update({"a": 1, "b": 2})
    assert d == {"a": 1, "b": 2}


def test_dict_clear() -> None:
    """clear() removes all items."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    d.clear()
    assert d == {}
    assert len(d) == 0
    d.clear()
    assert d == {}


def test_dict_copy() -> None:
    """copy() creates shallow copy."""
    original: dict[str, int] = {"a": 1, "b": 2}
    copied: dict[str, int] = original.copy()
    assert copied == original
    copied["c"] = 3
    assert copied == {"a": 1, "b": 2, "c": 3}
    assert original == {"a": 1, "b": 2}
    original["d"] = 4
    assert original == {"a": 1, "b": 2, "d": 4}
    assert copied == {"a": 1, "b": 2, "c": 3}


def test_dict_iteration_keys() -> None:
    """Iterating over dict yields keys."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    keys: list[str] = []
    for k in d:
        keys.append(k)
    assert len(keys) == 3
    assert "a" in keys
    assert "b" in keys
    assert "c" in keys


def test_dict_iteration_items() -> None:
    """Iterating over items()."""
    d: dict[str, int] = {"a": 1, "b": 2}
    result: dict[str, int] = {}
    for k, v in d.items():
        result[k] = v
    assert result == d


def test_dict_comprehension() -> None:
    """Dict comprehension."""
    squares: dict[int, int] = {x: x * x for x in [1, 2, 3, 4]}
    assert squares == {1: 1, 2: 4, 3: 9, 4: 16}


def test_dict_comprehension_condition() -> None:
    """Dict comprehension with condition."""
    evens: dict[int, int] = {x: x * x for x in [1, 2, 3, 4, 5, 6] if x % 2 == 0}
    assert evens == {2: 4, 4: 16, 6: 36}


def test_dict_int_keys() -> None:
    """Dict with int keys."""
    d: dict[int, str] = {1: "one", 2: "two", 3: "three"}
    assert d[1] == "one"
    assert d[2] == "two"
    assert 1 in d
    assert 4 not in d


def test_dict_mixed_operations() -> None:
    """Multiple operations on same dict."""
    d: dict[str, int] = {}
    d["a"] = 1
    d["b"] = 2
    assert len(d) == 2
    d["a"] = 10
    assert d["a"] == 10
    d["c"] = 3
    d["d"] = 4
    assert d == {"a": 10, "b": 2, "c": 3, "d": 4}


def test_dict_nested() -> None:
    """Nested dicts."""
    d: dict[str, dict[str, int]] = {
        "inner1": {"a": 1, "b": 2},
        "inner2": {"c": 3, "d": 4},
    }
    assert d["inner1"]["a"] == 1
    assert d["inner2"]["d"] == 4
    d["inner1"]["e"] = 5
    assert d["inner1"] == {"a": 1, "b": 2, "e": 5}


def test_dict_with_list_values() -> None:
    """Dict with list values."""
    d: dict[str, list[int]] = {"odds": [1, 3, 5], "evens": [2, 4, 6]}
    assert d["odds"] == [1, 3, 5]
    d["odds"].append(7)
    assert d["odds"] == [1, 3, 5, 7]


def test_dict_from_tuples() -> None:
    """dict() from list of tuples."""
    pairs: list[tuple[str, int]] = [("a", 1), ("b", 2), ("c", 3)]
    d: dict[str, int] = dict(pairs)
    assert d == {"a": 1, "b": 2, "c": 3}


def test_dict_from_zip() -> None:
    """dict() from zip."""
    keys: list[str] = ["a", "b", "c"]
    vals: list[int] = [1, 2, 3]
    d: dict[str, int] = dict(zip(keys, vals))
    assert d == {"a": 1, "b": 2, "c": 3}


def test_dict_identity() -> None:
    """Dict identity vs equality."""
    a: dict[str, int] = {"x": 1}
    b: dict[str, int] = {"x": 1}
    c: dict[str, int] = a
    assert a == b
    assert a == c
    a["y"] = 2
    assert c == {"x": 1, "y": 2}
    assert b == {"x": 1}


def test_dict_keys_view_membership() -> None:
    """Membership in keys view."""
    d: dict[str, int] = {"a": 1, "b": 2}
    assert "a" in d.keys()
    assert "c" not in d.keys()


def test_dict_values_sum() -> None:
    """sum() on values."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    assert sum(d.values()) == 6


def test_dict_empty_operations() -> None:
    """Operations on empty dict."""
    d: dict[str, int] = {}
    assert list(d.keys()) == []
    assert list(d.values()) == []
    assert list(d.items()) == []
    assert d.get("a") is None
    assert d.get("a", 5) == 5


def test_dict_overwrite() -> None:
    """Overwriting existing keys."""
    d: dict[str, int] = {"a": 1}
    d["a"] = 2
    assert d["a"] == 2
    d["a"] = 3
    assert d["a"] == 3
    assert len(d) == 1


def test_dict_bool_keys() -> None:
    """Dict with bool keys."""
    d: dict[bool, str] = {True: "yes", False: "no"}
    assert d[True] == "yes"
    assert d[False] == "no"
    assert len(d) == 2


def test_dict_tuple_keys() -> None:
    """Dict with tuple keys."""
    d: dict[tuple[int, int], str] = {(0, 0): "origin", (1, 0): "right", (0, 1): "up"}
    assert d[(0, 0)] == "origin"
    assert d[(1, 0)] == "right"
    assert (0, 0) in d
    assert (2, 2) not in d


def test_dict_none_value() -> None:
    """Dict with None values."""
    d: dict[str, int | None] = {"a": 1, "b": None, "c": 3}
    assert d["a"] == 1
    assert d["b"] is None
    assert d["c"] == 3
    assert "b" in d


def test_dict_update_from_empty() -> None:
    """update() from empty dict."""
    d: dict[str, int] = {"a": 1, "b": 2}
    d.update({})
    assert d == {"a": 1, "b": 2}


def test_dict_keys_sorted() -> None:
    """sorted() on keys."""
    d: dict[str, int] = {"c": 3, "a": 1, "b": 2}
    keys: list[str] = sorted(d.keys())
    assert keys == ["a", "b", "c"]


def test_dict_values_sorted() -> None:
    """sorted() on values."""
    d: dict[str, int] = {"c": 3, "a": 1, "b": 2}
    vals: list[int] = sorted(d.values())
    assert vals == [1, 2, 3]


def test_dict_min_max_keys() -> None:
    """min/max on keys."""
    d: dict[str, int] = {"c": 3, "a": 1, "b": 2}
    assert min(d) == "a"
    assert max(d) == "c"
    assert min(d.keys()) == "a"
    assert max(d.keys()) == "c"


def test_dict_min_max_values() -> None:
    """min/max on values."""
    d: dict[str, int] = {"c": 3, "a": 1, "b": 2}
    assert min(d.values()) == 1
    assert max(d.values()) == 3


def test_dict_int_float_key_equivalence() -> None:
    """Gotcha: int and float keys that are equal share the same slot."""
    d: dict[float, str] = {1: "int"}
    d[1.0] = "float"
    # 1 and 1.0 are equal and have same hash, so they're the same key
    assert len(d) == 1
    assert d[1] == "float"
    assert d[1.0] == "float"
    # Key stays as original type (1), value is updated
    keys: list[float] = list(d.keys())
    assert keys[0] == 1


def test_dict_bool_int_key_equivalence() -> None:
    """Gotcha: True==1 and False==0 as keys."""
    d: dict[int, str] = {True: "true", False: "false"}
    # True and False are treated as 1 and 0
    assert d[1] == "true"
    assert d[0] == "false"
    assert len(d) == 2
    # Setting with int overwrites bool key's value
    d[1] = "one"
    assert d[True] == "one"


def test_dict_copy_shallow_nested() -> None:
    """copy() is shallow - nested dicts are shared."""
    original: dict[str, dict[str, int]] = {"inner": {"a": 1}}
    copied: dict[str, dict[str, int]] = original.copy()
    # Outer dict is different
    copied["new"] = {"b": 2}
    assert "new" not in original
    # But inner dict is shared
    original["inner"]["a"] = 99
    assert copied["inner"]["a"] == 99


def test_dict_copy_shallow_list_values() -> None:
    """copy() is shallow - list values are shared."""
    original: dict[str, list[int]] = {"nums": [1, 2, 3]}
    copied: dict[str, list[int]] = original.copy()
    original["nums"].append(4)
    assert copied["nums"] == [1, 2, 3, 4]


def test_dict_setdefault_no_default() -> None:
    """setdefault() with no default uses None."""
    d: dict[str, int | None] = {"a": 1}
    x: int | None = d.setdefault("b")
    assert x is None
    assert d["b"] is None
    assert d == {"a": 1, "b": None}


def test_dict_setdefault_existing_unchanged() -> None:
    """setdefault() doesn't modify existing key."""
    d: dict[str, int] = {"a": 1}
    x: int = d.setdefault("a", 999)
    assert x == 1
    assert d["a"] == 1


def test_dict_popitem() -> None:
    """popitem() removes and returns last inserted item (LIFO)."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    item: tuple[str, int] = d.popitem()
    assert item == ("c", 3)
    assert d == {"a": 1, "b": 2}
    item = d.popitem()
    assert item == ("b", 2)
    assert d == {"a": 1}


def test_dict_insertion_order_preserved() -> None:
    """Insertion order is preserved (Python 3.7+)."""
    d: dict[str, int] = {}
    d["c"] = 3
    d["a"] = 1
    d["b"] = 2
    keys: list[str] = list(d.keys())
    assert keys == ["c", "a", "b"]
    vals: list[int] = list(d.values())
    assert vals == [3, 1, 2]


def test_dict_fromkeys() -> None:
    """dict.fromkeys() creates dict with same value for all keys."""
    keys: list[str] = ["a", "b", "c"]
    d: dict[str, int] = dict.fromkeys(keys, 0)
    assert d == {"a": 0, "b": 0, "c": 0}


def test_dict_fromkeys_default_none() -> None:
    """dict.fromkeys() without value uses None."""
    keys: list[str] = ["a", "b"]
    d: dict[str, None] = dict.fromkeys(keys)
    assert d == {"a": None, "b": None}


def test_dict_fromkeys_mutable_gotcha() -> None:
    """Gotcha: fromkeys() shares same mutable object."""
    keys: list[str] = ["a", "b", "c"]
    d: dict[str, list[int]] = dict.fromkeys(keys, [])
    d["a"].append(1)
    # All values are the same list object
    assert d["a"] == [1]
    assert d["b"] == [1]
    assert d["c"] == [1]


def test_dict_keys_view_set_operations() -> None:
    """Keys views support set operations."""
    d1: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    d2: dict[str, int] = {"b": 20, "c": 30, "d": 4}
    # Intersection
    common: set[str] = d1.keys() & d2.keys()
    assert common == {"b", "c"}
    # Union
    all_keys: set[str] = d1.keys() | d2.keys()
    assert all_keys == {"a", "b", "c", "d"}
    # Difference
    only_d1: set[str] = d1.keys() - d2.keys()
    assert only_d1 == {"a"}
    # Symmetric difference
    sym_diff: set[str] = d1.keys() ^ d2.keys()
    assert sym_diff == {"a", "d"}


def test_dict_items_view_set_operations() -> None:
    """Items views support set operations."""
    d1: dict[str, int] = {"a": 1, "b": 2}
    d2: dict[str, int] = {"a": 1, "b": 20}
    # Intersection - only items with same key AND value
    common: set[tuple[str, int]] = d1.items() & d2.items()
    assert common == {("a", 1)}


def test_dict_merge_operator() -> None:
    """Dict merge operator | creates new dict (Python 3.9+)."""
    d1: dict[str, int] = {"a": 1, "b": 2}
    d2: dict[str, int] = {"b": 20, "c": 3}
    merged: dict[str, int] = d1 | d2
    # Right operand wins for duplicate keys
    assert merged == {"a": 1, "b": 20, "c": 3}
    # Originals unchanged
    assert d1 == {"a": 1, "b": 2}
    assert d2 == {"b": 20, "c": 3}


def test_dict_update_operator() -> None:
    """Dict update operator |= modifies in place (Python 3.9+)."""
    d1: dict[str, int] = {"a": 1, "b": 2}
    d1 |= {"b": 20, "c": 3}
    assert d1 == {"a": 1, "b": 20, "c": 3}


def test_dict_merge_empty() -> None:
    """Merging with empty dicts."""
    d: dict[str, int] = {"a": 1}
    assert d | {} == {"a": 1}
    assert {} | d == {"a": 1}


def test_dict_update_overwrites() -> None:
    """update() overwrites values for existing keys."""
    d: dict[str, int] = {"a": 1, "b": 2}
    d.update({"a": 10, "b": 20})
    assert d == {"a": 10, "b": 20}


def test_dict_update_preserves_order() -> None:
    """update() preserves insertion order for existing keys."""
    d: dict[str, int] = {"a": 1, "b": 2, "c": 3}
    d.update({"b": 20})
    keys: list[str] = list(d.keys())
    # "b" stays in its original position
    assert keys == ["a", "b", "c"]


def test_dict_get_vs_index() -> None:
    """get() returns None for missing, [] raises KeyError."""
    d: dict[str, int] = {"a": 1}
    assert d.get("missing") is None
    assert d.get("missing", -1) == -1
    # d["missing"] would raise KeyError


def test_dict_len_after_modifications() -> None:
    """len() reflects current state after modifications."""
    d: dict[str, int] = {}
    assert len(d) == 0
    d["a"] = 1
    assert len(d) == 1
    d["a"] = 2  # Update, not insert
    assert len(d) == 1
    d["b"] = 2
    assert len(d) == 2


def test_dict_in_checks_keys_not_values() -> None:
    """'in' operator checks keys, not values."""
    d: dict[str, int] = {"key": 42}
    assert "key" in d
    assert 42 not in d  # 42 is a value, not a key


def test_dict_empty_string_key() -> None:
    """Empty string is a valid key."""
    d: dict[str, int] = {"": 0, "a": 1}
    assert d[""] == 0
    assert "" in d
    assert len(d) == 2


def test_dict_zero_key() -> None:
    """Zero is a valid key, distinct from empty/false."""
    d: dict[int, str] = {0: "zero", 1: "one"}
    assert d[0] == "zero"
    assert 0 in d


def test_dict_comprehension_overwrite() -> None:
    """Dict comprehension with duplicate keys keeps last."""
    items: list[tuple[str, int]] = [("a", 1), ("b", 2), ("a", 3)]
    d: dict[str, int] = {k: v for k, v in items}
    assert d["a"] == 3
    assert len(d) == 2


def test_dict_get_falsy_values() -> None:
    """get() with falsy values must return the actual value, not default."""
    d: dict[str, int] = {"a": 0, "b": 1}
    assert d.get("a", 99) == 0
    d2: dict[str, str] = {"x": "", "y": "hi"}
    assert d2.get("x", "default") == ""


def test_dict_any_all_items() -> None:
    """any/all with dict items tuple unpacking."""
    d: dict[str, int] = {"a": 1, "b": 20, "c": 3}
    assert any(v > 10 for k, v in d.items())
    assert all(v > 0 for k, v in d.items())
    assert not all(v > 10 for k, v in d.items())


def main() -> int:
    passed: int = 0
    failed: int = 0
    try:
        test_dict_equality()
        passed += 1
        print("  PASS test_dict_equality")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_equality: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_equality: " + str(e))
    try:
        test_dict_length()
        passed += 1
        print("  PASS test_dict_length")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_length: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_length: " + str(e))
    try:
        test_dict_get_item()
        passed += 1
        print("  PASS test_dict_get_item")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_get_item: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_get_item: " + str(e))
    try:
        test_dict_set_item()
        passed += 1
        print("  PASS test_dict_set_item")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_set_item: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_set_item: " + str(e))
    try:
        test_dict_set_item_empty()
        passed += 1
        print("  PASS test_dict_set_item_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_set_item_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_set_item_empty: " + str(e))
    try:
        test_dict_contains()
        passed += 1
        print("  PASS test_dict_contains")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_contains: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_contains: " + str(e))
    try:
        test_dict_contains_empty()
        passed += 1
        print("  PASS test_dict_contains_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_contains_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_contains_empty: " + str(e))
    try:
        test_dict_bool()
        passed += 1
        print("  PASS test_dict_bool")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_bool: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_bool: " + str(e))
    try:
        test_dict_get()
        passed += 1
        print("  PASS test_dict_get")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_get: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_get: " + str(e))
    try:
        test_dict_get_none_value()
        passed += 1
        print("  PASS test_dict_get_none_value")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_get_none_value: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_get_none_value: " + str(e))
    try:
        test_dict_keys()
        passed += 1
        print("  PASS test_dict_keys")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_keys: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_keys: " + str(e))
    try:
        test_dict_list_from_dict()
        passed += 1
        print("  PASS test_dict_list_from_dict")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_list_from_dict: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_list_from_dict: " + str(e))
    try:
        test_dict_set_from_dict()
        passed += 1
        print("  PASS test_dict_set_from_dict")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_set_from_dict: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_set_from_dict: " + str(e))
    try:
        test_dict_list_from_empty_dict()
        passed += 1
        print("  PASS test_dict_list_from_empty_dict")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_list_from_empty_dict: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_list_from_empty_dict: " + str(e))
    try:
        test_dict_set_from_empty_dict()
        passed += 1
        print("  PASS test_dict_set_from_empty_dict")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_set_from_empty_dict: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_set_from_empty_dict: " + str(e))
    try:
        test_dict_values()
        passed += 1
        print("  PASS test_dict_values")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_values: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_values: " + str(e))
    try:
        test_dict_items()
        passed += 1
        print("  PASS test_dict_items")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_items: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_items: " + str(e))
    try:
        test_dict_pop()
        passed += 1
        print("  PASS test_dict_pop")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_pop: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_pop: " + str(e))
    try:
        test_dict_pop_default()
        passed += 1
        print("  PASS test_dict_pop_default")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_pop_default: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_pop_default: " + str(e))
    try:
        test_dict_setdefault()
        passed += 1
        print("  PASS test_dict_setdefault")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_setdefault: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_setdefault: " + str(e))
    try:
        test_dict_update()
        passed += 1
        print("  PASS test_dict_update")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_update: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_update: " + str(e))
    try:
        test_dict_update_empty()
        passed += 1
        print("  PASS test_dict_update_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_update_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_update_empty: " + str(e))
    try:
        test_dict_clear()
        passed += 1
        print("  PASS test_dict_clear")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_clear: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_clear: " + str(e))
    try:
        test_dict_copy()
        passed += 1
        print("  PASS test_dict_copy")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_copy: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_copy: " + str(e))
    try:
        test_dict_iteration_keys()
        passed += 1
        print("  PASS test_dict_iteration_keys")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_iteration_keys: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_iteration_keys: " + str(e))
    try:
        test_dict_iteration_items()
        passed += 1
        print("  PASS test_dict_iteration_items")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_iteration_items: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_iteration_items: " + str(e))
    try:
        test_dict_comprehension()
        passed += 1
        print("  PASS test_dict_comprehension")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_comprehension: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_comprehension: " + str(e))
    try:
        test_dict_comprehension_condition()
        passed += 1
        print("  PASS test_dict_comprehension_condition")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_comprehension_condition: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_comprehension_condition: " + str(e))
    try:
        test_dict_int_keys()
        passed += 1
        print("  PASS test_dict_int_keys")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_int_keys: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_int_keys: " + str(e))
    try:
        test_dict_mixed_operations()
        passed += 1
        print("  PASS test_dict_mixed_operations")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_mixed_operations: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_mixed_operations: " + str(e))
    try:
        test_dict_nested()
        passed += 1
        print("  PASS test_dict_nested")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_nested: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_nested: " + str(e))
    try:
        test_dict_with_list_values()
        passed += 1
        print("  PASS test_dict_with_list_values")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_with_list_values: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_with_list_values: " + str(e))
    try:
        test_dict_from_tuples()
        passed += 1
        print("  PASS test_dict_from_tuples")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_from_tuples: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_from_tuples: " + str(e))
    try:
        test_dict_from_zip()
        passed += 1
        print("  PASS test_dict_from_zip")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_from_zip: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_from_zip: " + str(e))
    try:
        test_dict_identity()
        passed += 1
        print("  PASS test_dict_identity")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_identity: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_identity: " + str(e))
    try:
        test_dict_keys_view_membership()
        passed += 1
        print("  PASS test_dict_keys_view_membership")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_keys_view_membership: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_keys_view_membership: " + str(e))
    try:
        test_dict_values_sum()
        passed += 1
        print("  PASS test_dict_values_sum")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_values_sum: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_values_sum: " + str(e))
    try:
        test_dict_empty_operations()
        passed += 1
        print("  PASS test_dict_empty_operations")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_empty_operations: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_empty_operations: " + str(e))
    try:
        test_dict_overwrite()
        passed += 1
        print("  PASS test_dict_overwrite")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_overwrite: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_overwrite: " + str(e))
    try:
        test_dict_bool_keys()
        passed += 1
        print("  PASS test_dict_bool_keys")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_bool_keys: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_bool_keys: " + str(e))
    try:
        test_dict_tuple_keys()
        passed += 1
        print("  PASS test_dict_tuple_keys")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_tuple_keys: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_tuple_keys: " + str(e))
    try:
        test_dict_none_value()
        passed += 1
        print("  PASS test_dict_none_value")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_none_value: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_none_value: " + str(e))
    try:
        test_dict_update_from_empty()
        passed += 1
        print("  PASS test_dict_update_from_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_update_from_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_update_from_empty: " + str(e))
    try:
        test_dict_keys_sorted()
        passed += 1
        print("  PASS test_dict_keys_sorted")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_keys_sorted: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_keys_sorted: " + str(e))
    try:
        test_dict_values_sorted()
        passed += 1
        print("  PASS test_dict_values_sorted")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_values_sorted: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_values_sorted: " + str(e))
    try:
        test_dict_min_max_keys()
        passed += 1
        print("  PASS test_dict_min_max_keys")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_min_max_keys: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_min_max_keys: " + str(e))
    try:
        test_dict_min_max_values()
        passed += 1
        print("  PASS test_dict_min_max_values")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_min_max_values: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_min_max_values: " + str(e))
    try:
        test_dict_int_float_key_equivalence()
        passed += 1
        print("  PASS test_dict_int_float_key_equivalence")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_int_float_key_equivalence: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_int_float_key_equivalence: " + str(e))
    try:
        test_dict_bool_int_key_equivalence()
        passed += 1
        print("  PASS test_dict_bool_int_key_equivalence")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_bool_int_key_equivalence: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_bool_int_key_equivalence: " + str(e))
    try:
        test_dict_copy_shallow_nested()
        passed += 1
        print("  PASS test_dict_copy_shallow_nested")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_copy_shallow_nested: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_copy_shallow_nested: " + str(e))
    try:
        test_dict_copy_shallow_list_values()
        passed += 1
        print("  PASS test_dict_copy_shallow_list_values")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_copy_shallow_list_values: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_copy_shallow_list_values: " + str(e))
    try:
        test_dict_setdefault_no_default()
        passed += 1
        print("  PASS test_dict_setdefault_no_default")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_setdefault_no_default: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_setdefault_no_default: " + str(e))
    try:
        test_dict_popitem()
        passed += 1
        print("  PASS test_dict_popitem")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_popitem: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_popitem: " + str(e))
    try:
        test_dict_insertion_order_preserved()
        passed += 1
        print("  PASS test_dict_insertion_order_preserved")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_insertion_order_preserved: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_insertion_order_preserved: " + str(e))
    try:
        test_dict_fromkeys()
        passed += 1
        print("  PASS test_dict_fromkeys")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_fromkeys: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_fromkeys: " + str(e))
    try:
        test_dict_fromkeys_default_none()
        passed += 1
        print("  PASS test_dict_fromkeys_default_none")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_fromkeys_default_none: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_fromkeys_default_none: " + str(e))
    try:
        test_dict_fromkeys_mutable_gotcha()
        passed += 1
        print("  PASS test_dict_fromkeys_mutable_gotcha")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_fromkeys_mutable_gotcha: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_fromkeys_mutable_gotcha: " + str(e))
    try:
        test_dict_keys_view_set_operations()
        passed += 1
        print("  PASS test_dict_keys_view_set_operations")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_keys_view_set_operations: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_keys_view_set_operations: " + str(e))
    try:
        test_dict_items_view_set_operations()
        passed += 1
        print("  PASS test_dict_items_view_set_operations")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_items_view_set_operations: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_items_view_set_operations: " + str(e))
    try:
        test_dict_merge_operator()
        passed += 1
        print("  PASS test_dict_merge_operator")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_merge_operator: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_merge_operator: " + str(e))
    try:
        test_dict_update_operator()
        passed += 1
        print("  PASS test_dict_update_operator")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_update_operator: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_update_operator: " + str(e))
    try:
        test_dict_merge_empty()
        passed += 1
        print("  PASS test_dict_merge_empty")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_merge_empty: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_merge_empty: " + str(e))
    try:
        test_dict_update_overwrites()
        passed += 1
        print("  PASS test_dict_update_overwrites")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_update_overwrites: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_update_overwrites: " + str(e))
    try:
        test_dict_update_preserves_order()
        passed += 1
        print("  PASS test_dict_update_preserves_order")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_update_preserves_order: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_update_preserves_order: " + str(e))
    try:
        test_dict_get_vs_index()
        passed += 1
        print("  PASS test_dict_get_vs_index")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_get_vs_index: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_get_vs_index: " + str(e))
    try:
        test_dict_len_after_modifications()
        passed += 1
        print("  PASS test_dict_len_after_modifications")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_len_after_modifications: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_len_after_modifications: " + str(e))
    try:
        test_dict_in_checks_keys_not_values()
        passed += 1
        print("  PASS test_dict_in_checks_keys_not_values")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_in_checks_keys_not_values: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_in_checks_keys_not_values: " + str(e))
    try:
        test_dict_empty_string_key()
        passed += 1
        print("  PASS test_dict_empty_string_key")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_empty_string_key: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_empty_string_key: " + str(e))
    try:
        test_dict_zero_key()
        passed += 1
        print("  PASS test_dict_zero_key")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_zero_key: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_zero_key: " + str(e))
    try:
        test_dict_comprehension_overwrite()
        passed += 1
        print("  PASS test_dict_comprehension_overwrite")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_comprehension_overwrite: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_comprehension_overwrite: " + str(e))
    try:
        test_dict_get_falsy_values()
        passed += 1
        print("  PASS test_dict_get_falsy_values")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_get_falsy_values: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_get_falsy_values: " + str(e))
    try:
        test_dict_any_all_items()
        passed += 1
        print("  PASS test_dict_any_all_items")
    except AssertionError as e:
        failed += 1
        print("  FAIL test_dict_any_all_items: " + str(e))
    except Exception as e:
        failed += 1
        print("  FAIL test_dict_any_all_items: " + str(e))
    print(str(passed) + " passed, " + str(failed) + " failed")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
