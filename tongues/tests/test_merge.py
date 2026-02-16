"""Tests for Phase 3a: Project merge."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.frontend.parse import parse
from src.tongues import (
    _classify_import,
    _collect_module_names,
    _dependency_order,
    _detect_collisions,
    _parse_project_input,
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
        node = {"_type": "ImportFrom", "module": "foo", "level": 1, "names": []}
        assert _classify_import(node) == "project"

    def test_bare_relative_is_project(self):
        node = {"_type": "ImportFrom", "module": None, "level": 1, "names": []}
        assert _classify_import(node) == "project"

    def test_typing_is_stdlib(self):
        node = {"_type": "ImportFrom", "module": "typing", "level": 0, "names": []}
        assert _classify_import(node) == "stdlib"

    def test_dataclasses_is_stdlib(self):
        node = {"_type": "ImportFrom", "module": "dataclasses", "level": 0, "names": []}
        assert _classify_import(node) == "stdlib"

    def test_collections_abc_is_stdlib(self):
        node = {"_type": "ImportFrom", "module": "collections.abc", "level": 0, "names": []}
        assert _classify_import(node) == "stdlib"

    def test_future_is_stdlib(self):
        node = {"_type": "ImportFrom", "module": "__future__", "level": 0, "names": []}
        assert _classify_import(node) == "stdlib"

    def test_unknown_absolute_is_project(self):
        node = {"_type": "ImportFrom", "module": "mylib.utils", "level": 0, "names": []}
        assert _classify_import(node) == "project"

    def test_sys_is_stdlib(self):
        node = {"_type": "ImportFrom", "module": "sys", "level": 0, "names": []}
        assert _classify_import(node) == "stdlib"

    def test_os_is_stdlib(self):
        node = {"_type": "ImportFrom", "module": "os", "level": 0, "names": []}
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
        names = [{"name": "a", "asname": None}]
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
        deps = {"a.py": [], "b.py": ["a.py"], "c.py": ["a.py"], "d.py": ["b.py", "c.py"]}
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
        body = ast["body"]
        ann = body[1]
        assert ann["target"]["id"] == "Token" or ann["annotation"]["id"] == "Token"

    def test_no_match(self):
        ast = parse("x: int = 0\n")
        _rewrite_names(ast, {"Foo": "Bar"})
        assert ast["body"][0]["target"]["id"] == "x"


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
        assert merged["_type"] == "Module"
        body = merged["body"]
        names = [
            s.get("name", s.get("targets", [{}])[0].get("id", ""))
            for s in body
            if isinstance(s, dict) and s.get("_type") in ("FunctionDef", "ClassDef")
        ]
        assert "foo" in names
        assert "bar" in names

    def test_collision(self):
        file_asts = _parse_files(_load_fixture("collision"))
        merged, errors = merge_project(file_asts)
        assert merged is None
        assert len(errors) > 0
        assert any("Token" in e for e in errors)

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
            if isinstance(node, dict):
                if node.get("_type") == "Name" and node.get("id") == "Tok":
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
            if isinstance(node, dict):
                if node.get("_type") == "Attribute":
                    val = node.get("value", {})
                    if isinstance(val, dict) and val.get("id") == "parse":
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
        for stmt in merged["body"]:
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

    def test_collision_error(self):
        files = _load_fixture("collision")
        result = self._run(files, ["--stop-at", "subset"])
        assert result.returncode == 1
        assert b"duplicate name" in result.stderr

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
        file_names = {"defs.py": {"Token"}}
        errors = _rewrite_module_attrs(ast, module_bindings, file_names)
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
        file_names = {"defs.py": {"Token"}}
        errors = _rewrite_module_attrs(ast, module_bindings, file_names)
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
        result = self._run_bin(
            FIXTURES / "gather_test", ["--stop-at", "parse"]
        )
        output = result.stdout.decode()
        assert "hidden" not in output.lower() or "visible" in output
        # The hidden file defines 'hidden', it should NOT appear
        assert ".hidden_file.py" not in output

    def test_skips_pycache(self):
        """__pycache__ .py files should not be gathered."""
        result = self._run_bin(
            FIXTURES / "gather_test", ["--stop-at", "parse"]
        )
        output = result.stdout.decode()
        assert "cached" not in output

    def test_skips_tongues_skip(self):
        """Files with tongues: skip should not be gathered."""
        result = self._run_bin(
            FIXTURES / "gather_test", ["--stop-at", "parse"]
        )
        output = result.stdout.decode()
        assert "skipped" not in output

    def test_includes_visible(self):
        """Normal .py files should be gathered."""
        result = self._run_bin(
            FIXTURES / "gather_test", ["--stop-at", "parse"]
        )
        assert result.returncode == 0, result.stderr.decode()
        output = result.stdout.decode()
        assert "visible" in output

    def test_empty_dir(self):
        """Directory with no .py files should error."""
        result = self._run_bin(
            FIXTURES / "empty", ["--stop-at", "subset"]
        )
        assert result.returncode != 0
        assert b"no .py files" in result.stderr

    def test_skip_fixture_only_has_b(self):
        """The skip/ fixture should only gather b.py (a.py has tongues: skip)."""
        result = self._run_bin(
            FIXTURES / "skip", ["--stop-at", "parse"]
        )
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
            ("b.py", "from .a import foo\n\ndef bar(x: int) -> int:\n    return foo(x)\n"),
        ]
        # Should compile without error — pragma is recognized
        result = self._run(files, ["--target", "python"])
        assert result.returncode == 0, result.stderr.decode()

    def test_strict_tostring_pragma_propagates(self):
        """@@["strict_tostring"] in one file should enable strict tostring for the whole project."""
        files = [
            ("a.py", '@@["strict_tostring"]\ndef foo(x: int) -> int:\n    return x + 1\n'),
            ("b.py", "from .a import foo\n\ndef bar(x: int) -> int:\n    return foo(x)\n"),
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
