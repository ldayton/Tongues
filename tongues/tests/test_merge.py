"""Tests for Phase 3a: Project merge."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.frontend.parse import parse
from src.frontend.types import (
    JBool,
    JDict,
    JInt,
    JList,
    JNull,
    JStr,
    get_str,
    get_int,
    get_node,
    get_nodes,
)
from src.tongues import (
    _ast_equal,
    _classify_import,
    _collect_definition_refs,
    _collect_module_names,
    _compute_module_stems,
    _dependency_order,
    _detect_collisions,
    _parse_project_input,
    _plan_collision_resolution,
    _prefix_name,
    _resolve_project_import,
    _rewrite_names,
    _rewrite_module_attrs,
    merge_project,
    should_skip_file,
)

TONGUES_DIR = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "03a_merge" / "fixtures"


def _load_fixture(name: str) -> list[tuple[str, str]]:
    """Load a fixture directory into (relpath, source) pairs."""
    d = FIXTURES / name
    files = []
    for p in sorted(d.rglob("*.py")):
        rel = str(p.relative_to(d))
        source = p.read_text()
        if not should_skip_file(source):
            files.append((rel, source))
    return files


def _nul_encode(files: list[tuple[str, str]]) -> bytes:
    """Encode file list as NUL-delimited bytes for --project stdin."""
    parts = []
    for path, source in files:
        parts.append(path)
        parts.append(source)
    return "\0".join(parts).encode()


def _parse_files(files: list[tuple[str, str]]) -> list[tuple[str, dict]]:
    """Parse fixture files into (path, ast_dict) pairs."""
    return [(path, parse(source)) for path, source in files]


# ---------------------------------------------------------------------------
# Unit tests: _classify_import
# ---------------------------------------------------------------------------


class TestClassifyImport:
    def test_relative_is_project(self):
        node = {
            "_type": JStr("ImportFrom"),
            "module": JStr("foo"),
            "level": JInt(1),
            "names": JList([]),
        }
        assert _classify_import(node) == "project"

    def test_bare_relative_is_project(self):
        node = {
            "_type": JStr("ImportFrom"),
            "module": JNull(),
            "level": JInt(1),
            "names": JList([]),
        }
        assert _classify_import(node) == "project"

    def test_typing_is_stdlib(self):
        node = {
            "_type": JStr("ImportFrom"),
            "module": JStr("typing"),
            "level": JInt(0),
            "names": JList([]),
        }
        assert _classify_import(node) == "stdlib"

    def test_dataclasses_is_stdlib(self):
        node = {
            "_type": JStr("ImportFrom"),
            "module": JStr("dataclasses"),
            "level": JInt(0),
            "names": JList([]),
        }
        assert _classify_import(node) == "stdlib"

    def test_collections_abc_is_stdlib(self):
        node = {
            "_type": JStr("ImportFrom"),
            "module": JStr("collections.abc"),
            "level": JInt(0),
            "names": JList([]),
        }
        assert _classify_import(node) == "stdlib"

    def test_future_is_stdlib(self):
        node = {
            "_type": JStr("ImportFrom"),
            "module": JStr("__future__"),
            "level": JInt(0),
            "names": JList([]),
        }
        assert _classify_import(node) == "stdlib"

    def test_unknown_absolute_is_project(self):
        node = {
            "_type": JStr("ImportFrom"),
            "module": JStr("mylib.utils"),
            "level": JInt(0),
            "names": JList([]),
        }
        assert _classify_import(node) == "project"

    def test_sys_is_stdlib(self):
        node = {
            "_type": JStr("ImportFrom"),
            "module": JStr("sys"),
            "level": JInt(0),
            "names": JList([]),
        }
        assert _classify_import(node) == "stdlib"

    def test_os_is_stdlib(self):
        node = {
            "_type": JStr("ImportFrom"),
            "module": JStr("os"),
            "level": JInt(0),
            "names": JList([]),
        }
        assert _classify_import(node) == "stdlib"


# ---------------------------------------------------------------------------
# Unit tests: _resolve_project_import
# ---------------------------------------------------------------------------


class TestResolveProjectImport:
    def test_relative_module(self):
        universe = {"a.py", "b.py"}
        result = _resolve_project_import("b.py", "a", 1, [], universe)
        assert result == [("a.py", "")]

    def test_relative_bare_import(self):
        universe = {"a.py", "b.py"}
        names = [{"name": JStr("a"), "asname": JNull()}]
        result = _resolve_project_import("b.py", "", 1, names, universe)
        assert result == [("a.py", "")]

    def test_relative_in_subdir(self):
        universe = {"pkg/a.py", "pkg/b.py"}
        result = _resolve_project_import("pkg/b.py", "a", 1, [], universe)
        assert result == [("pkg/a.py", "")]

    def test_relative_up_level(self):
        universe = {"a.py", "sub/b.py"}
        result = _resolve_project_import("sub/b.py", "a", 2, [], universe)
        assert result == [("a.py", "")]

    def test_absolute_module(self):
        universe = {"pkg/mod.py"}
        result = _resolve_project_import("main.py", "pkg.mod", 0, [], universe)
        assert result == [("pkg/mod.py", "")]

    def test_init_fallback(self):
        universe = {"pkg/__init__.py"}
        result = _resolve_project_import("main.py", "pkg", 0, [], universe)
        assert result == [("pkg/__init__.py", "")]

    def test_unresolved(self):
        universe = {"a.py"}
        result = _resolve_project_import("a.py", "missing", 1, [], universe)
        assert len(result) == 1
        assert result[0][0] == ""
        assert "unresolved" in result[0][1]

    def test_dotted_relative(self):
        universe = {"pkg/sub/mod.py", "main.py"}
        result = _resolve_project_import("main.py", "pkg.sub.mod", 1, [], universe)
        assert result == [("pkg/sub/mod.py", "")]


# ---------------------------------------------------------------------------
# Unit tests: _dependency_order
# ---------------------------------------------------------------------------


class TestDependencyOrder:
    def test_simple_dag(self):
        files = ["a.py", "b.py"]
        deps = {"a.py": [], "b.py": ["a.py"]}
        result = _dependency_order(files, deps)
        assert result.index("b.py") < result.index("a.py")

    def test_no_deps(self):
        files = ["c.py", "a.py", "b.py"]
        result = _dependency_order(files, {})
        assert result == ["a.py", "b.py", "c.py"]

    def test_cycle(self):
        files = ["a.py", "b.py"]
        deps = {"a.py": ["b.py"], "b.py": ["a.py"]}
        result = _dependency_order(files, deps)
        assert set(result) == {"a.py", "b.py"}
        assert len(result) == 2

    def test_single_file(self):
        assert _dependency_order(["a.py"], {}) == ["a.py"]

    def test_diamond(self):
        files = ["a.py", "b.py", "c.py", "d.py"]
        deps = {
            "a.py": [],
            "b.py": ["a.py"],
            "c.py": ["a.py"],
            "d.py": ["b.py", "c.py"],
        }
        result = _dependency_order(files, deps)
        assert result.index("d.py") < result.index("b.py")
        assert result.index("d.py") < result.index("c.py")
        assert result.index("b.py") < result.index("a.py")
        assert result.index("c.py") < result.index("a.py")


# ---------------------------------------------------------------------------
# Unit tests: _collect_module_names
# ---------------------------------------------------------------------------


class TestCollectModuleNames:
    def test_classdef(self):
        ast = parse("class Foo:\n    pass\n")
        names = _collect_module_names(ast)
        assert [n[0] for n in names] == ["Foo"]

    def test_funcdef(self):
        ast = parse("def foo(x: int) -> int:\n    return x\n")
        names = _collect_module_names(ast)
        assert [n[0] for n in names] == ["foo"]

    def test_assign(self):
        ast = parse("x: int = 0\n")
        names = _collect_module_names(ast)
        assert [n[0] for n in names] == ["x"]

    def test_import_not_collected(self):
        ast = parse("from typing import List\n")
        names = _collect_module_names(ast)
        assert names == []

    def test_multiple(self):
        source = "class A:\n    pass\ndef b(x: int) -> int:\n    return x\nC: int = 1\n"
        ast = parse(source)
        names = _collect_module_names(ast)
        assert [n[0] for n in names] == ["A", "b", "C"]


# ---------------------------------------------------------------------------
# Unit tests: _detect_collisions
# ---------------------------------------------------------------------------


class TestDetectCollisions:
    def test_no_collision(self):
        file_names = {
            "a.py": [("foo", 1, 0)],
            "b.py": [("bar", 1, 0)],
        }
        assert _detect_collisions(file_names) == []

    def test_collision(self):
        file_names = {
            "a.py": [("Token", 1, 0)],
            "b.py": [("Token", 3, 0)],
        }
        errors = _detect_collisions(file_names)
        assert len(errors) == 1
        assert "Token" in errors[0]
        assert "a.py" in errors[0]
        assert "b.py" in errors[0]

    def test_multiple_collisions(self):
        file_names = {
            "a.py": [("X", 1, 0), ("Y", 2, 0)],
            "b.py": [("X", 1, 0), ("Y", 2, 0)],
        }
        errors = _detect_collisions(file_names)
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# Unit tests: _rewrite_names
# ---------------------------------------------------------------------------


class TestRewriteNames:
    def test_simple_rename(self):
        ast = parse("from .a import Token as Tok\nx: Tok = Tok('hi')\n")
        _rewrite_names(ast, {"Tok": "Token"})
        body = get_nodes(ast, "body")
        ann = body[1]
        assert (
            get_str(ann, "id") == "Token"
            or get_str(get_node(ann, "target"), "id") == "Token"
            or get_str(get_node(ann, "annotation"), "id") == "Token"
        )

    def test_no_match(self):
        ast = parse("x: int = 0\n")
        _rewrite_names(ast, {"Foo": "Bar"})
        body = get_nodes(ast, "body")
        assert get_str(get_node(body[0], "target"), "id") == "x"


# ---------------------------------------------------------------------------
# Unit tests: _parse_project_input
# ---------------------------------------------------------------------------


class TestParseProjectInput:
    def test_basic(self):
        data = "a.py\0def foo(): pass\0b.py\0def bar(): pass\0"
        result = _parse_project_input(data)
        assert len(result) == 2
        assert result[0] == ("a.py", "def foo(): pass")
        assert result[1] == ("b.py", "def bar(): pass")

    def test_empty(self):
        assert _parse_project_input("") == []

    def test_single_file(self):
        data = "a.py\0content\0"
        result = _parse_project_input(data)
        assert result == [("a.py", "content")]


# ---------------------------------------------------------------------------
# Integration: merge_project
# ---------------------------------------------------------------------------


class TestMergeProject:
    def test_basic(self):
        file_asts = _parse_files(_load_fixture("basic"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None
        assert get_str(merged, "_type") == "Module"
        body = get_nodes(merged, "body")
        names = [
            get_str(s, "name") or get_str(get_nodes(s, "targets")[0], "id")
            if get_nodes(s, "targets")
            else get_str(s, "name")
            for s in body
            if get_str(s, "_type") in ("FunctionDef", "ClassDef")
        ]
        assert "foo" in names
        assert "bar" in names

    def test_collision_resolved(self):
        file_asts = _parse_files(_load_fixture("collision"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None
        # Both Token classes should be prefixed
        names = []
        for s in get_nodes(merged, "body"):
            if get_str(s, "_type") == "ClassDef":
                names.append(get_str(s, "name"))
        assert "a_Token" in names
        assert "b_Token" in names
        assert "Token" not in names

    def test_alias_rewriting(self):
        file_asts = _parse_files(_load_fixture("alias"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None
        # After merge, 'Tok' references should be rewritten to 'Token'
        # Check that no Name node with id='Tok' remains
        found_tok = False

        def walk(node):
            nonlocal found_tok
            if isinstance(node, JDict):
                walk(node.entries)
            elif isinstance(node, JList):
                for item in node.items:
                    walk(item)
            elif isinstance(node, dict):
                if get_str(node, "_type") == "Name" and get_str(node, "id") == "Tok":
                    found_tok = True
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(merged)
        assert not found_tok, "Found unrewritten 'Tok' reference"

    def test_module_attr_rewriting(self):
        file_asts = _parse_files(_load_fixture("module_name"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None
        # After merge, parse.Token should be rewritten to Token
        found_attr = False

        def walk(node):
            nonlocal found_attr
            if isinstance(node, JDict):
                walk(node.entries)
            elif isinstance(node, JList):
                for item in node.items:
                    walk(item)
            elif isinstance(node, dict):
                if get_str(node, "_type") == "Attribute":
                    val = node.get("value")
                    if isinstance(val, JDict) and get_str(val.entries, "id") == "parse":
                        found_attr = True
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(merged)
        assert not found_attr, "Found unrewritten 'parse.X' attribute access"

    def test_cycle(self):
        file_asts = _parse_files(_load_fixture("cycle"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None

    def test_unresolved(self):
        file_asts = _parse_files(_load_fixture("unresolved"))
        merged, errors = merge_project(file_asts)
        assert merged is None
        assert len(errors) > 0
        assert any("unresolved" in e for e in errors)

    def test_nested_package(self):
        file_asts = _parse_files(_load_fixture("nested"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None

    def test_source_file_tags(self):
        file_asts = _parse_files(_load_fixture("basic"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        for stmt in get_nodes(merged, "body"):
            assert "_source_file" in stmt


# ---------------------------------------------------------------------------
# CLI integration: --project flag via subprocess
# ---------------------------------------------------------------------------


class TestProjectCLI:
    def _run(self, files, extra_args=None):
        args = [sys.executable, "-m", "src.tongues", "--project"]
        if extra_args:
            args.extend(extra_args)
        stdin_data = _nul_encode(files)
        return subprocess.run(
            args, input=stdin_data, capture_output=True, cwd=TONGUES_DIR
        )

    def test_basic_subset(self):
        files = _load_fixture("basic")
        result = self._run(files, ["--stop-at", "subset"])
        assert result.returncode == 0, result.stderr.decode()

    def test_basic_names(self):
        files = _load_fixture("basic")
        result = self._run(files, ["--stop-at", "names"])
        assert result.returncode == 0, result.stderr.decode()

    def test_collision_resolved(self):
        files = _load_fixture("collision")
        result = self._run(files, ["--stop-at", "subset"])
        assert result.returncode == 0, result.stderr.decode()

    def test_unresolved_error(self):
        files = _load_fixture("unresolved")
        result = self._run(files, ["--stop-at", "subset"])
        assert result.returncode == 1
        assert b"unresolved" in result.stderr

    def test_empty_input(self):
        result = self._run([], ["--stop-at", "subset"])
        assert result.returncode != 0

    def test_parse_output_json(self):
        files = _load_fixture("basic")
        result = self._run(files, ["--stop-at", "parse"])
        assert result.returncode == 0
        output = result.stdout.decode()
        assert '"path"' in output
        assert '"ast"' in output

    def test_basic_python_emit(self):
        files = _load_fixture("basic")
        result = self._run(files, ["--target", "python"])
        assert result.returncode == 0, result.stderr.decode()
        output = result.stdout.decode()
        assert "foo" in output
        assert "bar" in output

    def test_alias_python_emit(self):
        files = _load_fixture("alias")
        result = self._run(files, ["--target", "python"])
        assert result.returncode == 0, result.stderr.decode()

    def test_cycle_python_emit(self):
        files = _load_fixture("cycle")
        result = self._run(files, ["--target", "python"])
        assert result.returncode == 0, result.stderr.decode()

    def test_nested_python_emit(self):
        files = _load_fixture("nested")
        result = self._run(files, ["--target", "python"])
        assert result.returncode == 0, result.stderr.decode()


# ---------------------------------------------------------------------------
# Unit tests: _rewrite_module_attrs error path
# ---------------------------------------------------------------------------


class TestRewriteModuleAttrsError:
    def test_unknown_attr_error(self):
        """module.Nonexistent should produce an error when attr not in target file's names."""
        ast = parse(
            "from . import defs\n"
            "\n"
            "def make() -> defs.Nonexistent:\n"
            "    return defs.Nonexistent()\n"
        )
        module_bindings = {"defs": "defs.py"}
        file_name_map = {"defs.py": {"Token": "Token"}}
        errors = _rewrite_module_attrs(ast, module_bindings, file_name_map)
        assert len(errors) > 0
        assert any("Nonexistent" in e for e in errors)
        assert any("defs.py" in e for e in errors)

    def test_valid_attr_no_error(self):
        """module.Token should rewrite cleanly when Token exists in target."""
        ast = parse(
            "from . import defs\n"
            "\n"
            "def make() -> defs.Token:\n"
            "    return defs.Token('x')\n"
        )
        module_bindings = {"defs": "defs.py"}
        file_name_map = {"defs.py": {"Token": "Token"}}
        errors = _rewrite_module_attrs(ast, module_bindings, file_name_map)
        assert errors == []

    def test_bad_attr_via_merge(self):
        """End-to-end: merge_project should report error for nonexistent attr."""
        file_asts = _parse_files(_load_fixture("bad_attr"))
        merged, errors = merge_project(file_asts)
        assert merged is None
        assert len(errors) > 0
        assert any("Nonexistent" in e for e in errors)


# ---------------------------------------------------------------------------
# gather_project_files via bin/tongues subprocess
# ---------------------------------------------------------------------------


class TestGatherProjectFiles:
    def _run_bin(self, fixture_dir, extra_args=None):
        args = [sys.executable, str(TONGUES_DIR / "bin" / "tongues")]
        if extra_args:
            args.extend(extra_args)
        args.append(str(fixture_dir))
        return subprocess.run(args, capture_output=True, cwd=TONGUES_DIR)

    def test_skips_hidden_files(self):
        """Hidden .py files should not be gathered."""
        result = self._run_bin(FIXTURES / "gather_test", ["--stop-at", "parse"])
        output = result.stdout.decode()
        assert "hidden" not in output.lower() or "visible" in output
        # The hidden file defines 'hidden', it should NOT appear
        assert ".hidden_file.py" not in output

    def test_skips_pycache(self):
        """__pycache__ .py files should not be gathered."""
        result = self._run_bin(FIXTURES / "gather_test", ["--stop-at", "parse"])
        output = result.stdout.decode()
        assert "cached" not in output

    def test_skips_tongues_skip(self):
        """Files with tongues: skip should not be gathered."""
        result = self._run_bin(FIXTURES / "gather_test", ["--stop-at", "parse"])
        output = result.stdout.decode()
        assert "skipped" not in output

    def test_includes_visible(self):
        """Normal .py files should be gathered."""
        result = self._run_bin(FIXTURES / "gather_test", ["--stop-at", "parse"])
        assert result.returncode == 0, result.stderr.decode()
        output = result.stdout.decode()
        assert "visible" in output

    def test_empty_dir(self):
        """Directory with no .py files should error."""
        result = self._run_bin(FIXTURES / "empty", ["--stop-at", "subset"])
        assert result.returncode != 0
        assert b"no .py files" in result.stderr

    def test_skip_fixture_only_has_b(self):
        """The skip/ fixture should only gather b.py (a.py has tongues: skip)."""
        result = self._run_bin(FIXTURES / "skip", ["--stop-at", "parse"])
        assert result.returncode == 0, result.stderr.decode()
        output = result.stdout.decode()
        assert "b.py" in output
        # a.py should be skipped entirely — its content should not appear
        assert "requests" not in output


# ---------------------------------------------------------------------------
# _source_file in error output
# ---------------------------------------------------------------------------


class TestSourceFileErrorFormat:
    def _run(self, files, extra_args=None):
        args = [sys.executable, "-m", "src.tongues", "--project"]
        if extra_args:
            args.extend(extra_args)
        stdin_data = _nul_encode(files)
        return subprocess.run(
            args, input=stdin_data, capture_output=True, cwd=TONGUES_DIR
        )

    def test_subset_error_includes_source_file(self):
        """Subset violation in merged project should include the source filename."""
        files = _load_fixture("subset_error")
        result = self._run(files, ["--stop-at", "subset"])
        assert result.returncode == 1
        stderr = result.stderr.decode()
        # b.py has unannotated param — error should mention b.py
        assert "b.py:" in stderr

    def test_names_error_includes_source_file(self):
        """Name resolution error in merged project should include source filename."""
        # Construct inline: b.py references undefined name
        files = [
            ("a.py", "def foo(x: int) -> int:\n    return x\n"),
            ("b.py", "from .a import foo\n\ndef bar() -> int:\n    return baz()\n"),
        ]
        result = self._run(files, ["--stop-at", "names"])
        assert result.returncode == 1
        stderr = result.stderr.decode()
        assert "b.py:" in stderr


# ---------------------------------------------------------------------------
# --project with pragma interaction
# ---------------------------------------------------------------------------


class TestProjectPragmas:
    def _run(self, files, extra_args=None):
        args = [sys.executable, "-m", "src.tongues", "--project"]
        if extra_args:
            args.extend(extra_args)
        stdin_data = _nul_encode(files)
        return subprocess.run(
            args, input=stdin_data, capture_output=True, cwd=TONGUES_DIR
        )

    def test_strict_math_pragma_propagates(self):
        """@@["strict_math"] in one file should enable strict math for the whole project."""
        files = [
            ("a.py", '@@["strict_math"]\ndef foo(x: int) -> int:\n    return x + 1\n'),
            (
                "b.py",
                "from .a import foo\n\ndef bar(x: int) -> int:\n    return foo(x)\n",
            ),
        ]
        # Should compile without error — pragma is recognized
        result = self._run(files, ["--target", "python"])
        assert result.returncode == 0, result.stderr.decode()

    def test_strict_tostring_pragma_propagates(self):
        """@@["strict_tostring"] in one file should enable strict tostring for the whole project."""
        files = [
            (
                "a.py",
                '@@["strict_tostring"]\ndef foo(x: int) -> int:\n    return x + 1\n',
            ),
            (
                "b.py",
                "from .a import foo\n\ndef bar(x: int) -> int:\n    return foo(x)\n",
            ),
        ]
        result = self._run(files, ["--target", "python"])
        assert result.returncode == 0, result.stderr.decode()

    def test_strict_flag_with_project(self):
        """--strict flag should work with --project."""
        files = [
            ("a.py", "def foo(x: int) -> int:\n    return x + 1\n"),
        ]
        result = self._run(files, ["--target", "python", "--strict"])
        assert result.returncode == 0, result.stderr.decode()


# ---------------------------------------------------------------------------
# Unit tests: _ast_equal
# ---------------------------------------------------------------------------


class TestAstEqual:
    def test_identical_dicts(self):
        a = {"_type": "Name", "id": "foo", "lineno": 1}
        b = {"_type": "Name", "id": "foo", "lineno": 5}
        assert _ast_equal(a, b)

    def test_differ_in_value(self):
        a = {"_type": "Name", "id": "foo"}
        b = {"_type": "Name", "id": "bar"}
        assert not _ast_equal(a, b)

    def test_differ_in_position_still_equal(self):
        a = {"_type": "Constant", "value": 1, "lineno": 1, "col_offset": 0}
        b = {"_type": "Constant", "value": 1, "lineno": 99, "col_offset": 42}
        assert _ast_equal(a, b)

    def test_nested_dicts(self):
        a = {"_type": "Assign", "value": {"_type": "Constant", "value": 1, "lineno": 1}}
        b = {"_type": "Assign", "value": {"_type": "Constant", "value": 1, "lineno": 9}}
        assert _ast_equal(a, b)

    def test_nested_differ(self):
        a = {"_type": "Assign", "value": {"_type": "Constant", "value": 1}}
        b = {"_type": "Assign", "value": {"_type": "Constant", "value": 2}}
        assert not _ast_equal(a, b)

    def test_lists_equal(self):
        a = {"_type": "Module", "body": [{"_type": "Pass"}, {"_type": "Pass"}]}
        b = {"_type": "Module", "body": [{"_type": "Pass"}, {"_type": "Pass"}]}
        assert _ast_equal(a, b)

    def test_lists_differ_length(self):
        a = {"_type": "Module", "body": [{"_type": "Pass"}]}
        b = {"_type": "Module", "body": [{"_type": "Pass"}, {"_type": "Pass"}]}
        assert not _ast_equal(a, b)

    def test_source_file_ignored(self):
        a = {"_type": "Name", "id": "x", "_source_file": "a.py"}
        b = {"_type": "Name", "id": "x", "_source_file": "b.py"}
        assert _ast_equal(a, b)

    def test_different_keys(self):
        a = {"_type": "Name", "id": "x"}
        b = {"_type": "Name", "id": "x", "extra": True}
        assert not _ast_equal(a, b)


# ---------------------------------------------------------------------------
# Unit tests: _collect_definition_refs
# ---------------------------------------------------------------------------


class TestCollectDefinitionRefs:
    def test_function_with_refs(self):
        ast = parse("def foo(x: int) -> int:\n    return bar(x)\n")
        func = get_nodes(ast, "body")[0]
        refs = _collect_definition_refs(func)
        assert "bar" in refs
        assert "int" in refs

    def test_simple_constant(self):
        ast = parse("X: int = 42\n")
        stmt = get_nodes(ast, "body")[0]
        refs = _collect_definition_refs(stmt)
        assert "int" in refs

    def test_class_with_bases(self):
        ast = parse("class Foo(Base):\n    def method(self) -> None:\n        pass\n")
        cls = get_nodes(ast, "body")[0]
        refs = _collect_definition_refs(cls)
        assert "Base" in refs

    def test_assign_value_refs(self):
        ast = parse("x: list[str] = make_list()\n")
        stmt = get_nodes(ast, "body")[0]
        refs = _collect_definition_refs(stmt)
        assert "make_list" in refs
        assert "list" in refs


# ---------------------------------------------------------------------------
# Unit tests: _compute_module_stems
# ---------------------------------------------------------------------------


class TestComputeModuleStems:
    def test_unique_stems(self):
        paths = ["a.py", "b.py", "c.py"]
        stems = _compute_module_stems(paths)
        assert stems == {"a.py": "a", "b.py": "b", "c.py": "c"}

    def test_conflicting_stems(self):
        paths = ["frontend/parse.py", "taytsh/parse.py"]
        stems = _compute_module_stems(paths)
        assert stems["frontend/parse.py"] == "frontend_parse"
        assert stems["taytsh/parse.py"] == "taytsh_parse"

    def test_init_py(self):
        paths = ["pkg/__init__.py", "other.py"]
        stems = _compute_module_stems(paths)
        assert stems["pkg/__init__.py"] == "pkg"
        assert stems["other.py"] == "other"

    def test_init_conflict(self):
        paths = ["pkg/__init__.py", "pkg.py"]
        stems = _compute_module_stems(paths)
        # Both would be "pkg" — should be disambiguated
        vals = list(stems.values())
        assert len(set(vals)) == 2

    def test_no_extension(self):
        paths = ["noext"]
        stems = _compute_module_stems(paths)
        assert stems["noext"] == "noext"


# ---------------------------------------------------------------------------
# Unit tests: _prefix_name
# ---------------------------------------------------------------------------


class TestPrefixName:
    def test_private(self):
        assert _prefix_name("_helper", "scope") == "_scope_helper"

    def test_public(self):
        assert _prefix_name("Token", "parse") == "parse_Token"

    def test_dunder_private(self):
        assert _prefix_name("__private", "mod") == "_mod__private"

    def test_all_caps_public(self):
        assert _prefix_name("TY_ERROR", "runtime") == "RUNTIME_TY_ERROR"

    def test_all_caps_private(self):
        assert _prefix_name("_MAX_SIZE", "util") == "_UTIL_MAX_SIZE"

    def test_single_letter_caps(self):
        assert _prefix_name("X", "a") == "A_X"


# ---------------------------------------------------------------------------
# Unit tests: _plan_collision_resolution
# ---------------------------------------------------------------------------


class TestPlanCollisionResolution:
    def _make_names(self, source):
        """Parse source and return _collect_module_names result."""
        return _collect_module_names(parse(source))

    def test_all_identical_dedup(self):
        a_names = self._make_names("X: int = 1\n")
        b_names = self._make_names("X: int = 1\n")
        file_names = {"a.py": a_names, "b.py": b_names}
        stems = {"a.py": "a", "b.py": "b"}
        dedup, renames = _plan_collision_resolution(file_names, stems)
        assert "X" in dedup
        assert renames == {} or all("X" not in v for v in renames.values())

    def test_all_different_prefix(self):
        a_names = self._make_names("X: int = 1\n")
        b_names = self._make_names("X: int = 2\n")
        file_names = {"a.py": a_names, "b.py": b_names}
        stems = {"a.py": "a", "b.py": "b"}
        dedup, renames = _plan_collision_resolution(file_names, stems)
        assert "X" not in dedup
        assert renames["a.py"]["X"] == "A_X"
        assert renames["b.py"]["X"] == "B_X"

    def test_no_collision(self):
        a_names = self._make_names("X: int = 1\n")
        b_names = self._make_names("Y: int = 2\n")
        file_names = {"a.py": a_names, "b.py": b_names}
        stems = {"a.py": "a", "b.py": "b"}
        dedup, renames = _plan_collision_resolution(file_names, stems)
        assert len(dedup) == 0
        assert renames == {}

    def test_unsafe_dedup_demoted(self):
        a_names = self._make_names(
            "def _helper() -> int:\n    return 1\n"
            "\n"
            "def wrapper() -> int:\n    return _helper()\n"
        )
        b_names = self._make_names(
            "def _helper() -> int:\n    return 2\n"
            "\n"
            "def wrapper() -> int:\n    return _helper()\n"
        )
        file_names = {"a.py": a_names, "b.py": b_names}
        stems = {"a.py": "a", "b.py": "b"}
        dedup, renames = _plan_collision_resolution(file_names, stems)
        # _helper is different → prefixed
        assert "_helper" not in dedup
        # wrapper is identical but refs _helper which is prefixed → demoted
        assert "wrapper" not in dedup
        assert "wrapper" in renames.get("a.py", {})
        assert "wrapper" in renames.get("b.py", {})


# ---------------------------------------------------------------------------
# Integration: collision resolution merges
# ---------------------------------------------------------------------------


class TestPrefixedMerge:
    def test_private_prefix(self):
        """Two files with same private fn get prefixed."""
        file_asts = _parse_files(_load_fixture("prefix_private"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None
        names = [
            get_str(s, "name")
            for s in get_nodes(merged, "body")
            if get_str(s, "_type") == "FunctionDef"
        ]
        assert "_a_helper" in names
        assert "_b_helper" in names
        assert "_helper" not in names
        assert "use_a" in names
        assert "use_b" in names

    def test_private_refs_updated(self):
        """References to _helper in each file are updated to the prefixed version."""
        file_asts = _parse_files(_load_fixture("prefix_private"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        # Collect all Name.id references
        name_ids = set()

        def walk(node):
            if isinstance(node, JDict):
                walk(node.entries)
            elif isinstance(node, JList):
                for item in node.items:
                    walk(item)
            elif isinstance(node, dict):
                if get_str(node, "_type") == "Name":
                    name_ids.add(get_str(node, "id"))
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(merged)
        assert "_helper" not in name_ids
        assert "_a_helper" in name_ids
        assert "_b_helper" in name_ids


class TestDedupMerge:
    def test_dedup_keeps_one_copy(self):
        """Two files with identical constant produce one copy."""
        file_asts = _parse_files(_load_fixture("dedup"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None
        # Count occurrences of ASTNode definition (Assign, not AnnAssign)
        count = 0
        for s in get_nodes(merged, "body"):
            if get_str(s, "_type") == "Assign":
                targets = get_nodes(s, "targets")
                if len(targets) > 0:
                    t = targets[0]
                    if get_str(t, "id") == "ASTNode":
                        count += 1
        assert count == 1

    def test_dedup_both_functions_present(self):
        file_asts = _parse_files(_load_fixture("dedup"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        names = [
            get_str(s, "name")
            for s in get_nodes(merged, "body")
            if get_str(s, "_type") == "FunctionDef"
        ]
        assert "make_a" in names
        assert "make_b" in names


class TestPrefixedImport:
    def test_import_refs_updated(self):
        """File importing a prefixed name gets its references updated."""
        file_asts = _parse_files(_load_fixture("prefix_import"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None
        # LibFoo alias should resolve to lib_Foo (the prefixed name)
        name_ids = set()

        def walk(node):
            if isinstance(node, JDict):
                walk(node.entries)
            elif isinstance(node, JList):
                for item in node.items:
                    walk(item)
            elif isinstance(node, dict):
                if get_str(node, "_type") == "Name":
                    name_ids.add(get_str(node, "id"))
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(merged)
        assert "lib_Foo" in name_ids
        assert "LibFoo" not in name_ids
        assert "Foo" not in name_ids

    def test_both_classes_present(self):
        file_asts = _parse_files(_load_fixture("prefix_import"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        class_names = [
            get_str(s, "name")
            for s in get_nodes(merged, "body")
            if get_str(s, "_type") == "ClassDef"
        ]
        assert "lib_Foo" in class_names
        assert "app_Foo" in class_names


class TestPrefixedModuleAttr:
    def test_module_attr_uses_prefixed_name(self):
        """from . import defs; defs.Token uses the prefixed name."""
        file_asts = _parse_files(_load_fixture("prefix_module_attr"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None
        # After merge, defs.Token should be rewritten to defs_Token
        name_ids = set()

        def walk(node):
            if isinstance(node, JDict):
                walk(node.entries)
            elif isinstance(node, JList):
                for item in node.items:
                    walk(item)
            elif isinstance(node, dict):
                if get_str(node, "_type") == "Name":
                    name_ids.add(get_str(node, "id"))
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(merged)
        assert "defs_Token" in name_ids
        # No unrewritten module.attr references should remain
        found_attr = False

        def walk_attr(node):
            nonlocal found_attr
            if isinstance(node, JDict):
                walk_attr(node.entries)
            elif isinstance(node, JList):
                for item in node.items:
                    walk_attr(item)
            elif isinstance(node, dict):
                if get_str(node, "_type") == "Attribute":
                    val = node.get("value")
                    if isinstance(val, JDict) and get_str(val.entries, "id") == "defs":
                        found_attr = True
                for v in node.values():
                    walk_attr(v)
            elif isinstance(node, list):
                for item in node:
                    walk_attr(item)

        walk_attr(merged)
        assert not found_attr


class TestUnsafeDedupMerge:
    def test_wrapper_demoted_to_prefix(self):
        """Identical wrapper() calling different _helper() gets prefixed."""
        file_asts = _parse_files(_load_fixture("unsafe_dedup"))
        merged, errors = merge_project(file_asts)
        assert errors == []
        assert merged is not None
        names = [
            get_str(s, "name")
            for s in get_nodes(merged, "body")
            if get_str(s, "_type") == "FunctionDef"
        ]
        assert "_a_helper" in names
        assert "_b_helper" in names
        assert "a_wrapper" in names
        assert "b_wrapper" in names
        assert "_helper" not in names
        assert "wrapper" not in names
