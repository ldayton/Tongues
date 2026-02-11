"""Taytsh parser and typechecker — public API."""

from __future__ import annotations

from .ast import TModule
from .check import CheckError, check as check_module
from .emit import to_source
from .parse import ParseError as ParseError, Parser
from .tokens import TokenizeError as TokenizeError, tokenize


def _extract_pragmas(source: str) -> tuple[bool, bool]:
    """Scan leading lines for pragma comments. Returns (strict_math, strict_tostring)."""
    strict_math = False
    strict_tostring = False
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped == "":
            continue
        if not stripped.startswith("--"):
            break
        body = stripped[2:].strip()
        if body == "pragma strict-math":
            strict_math = True
        elif body == "pragma strict-tostring":
            strict_tostring = True
    return strict_math, strict_tostring


def parse(source: str) -> TModule:
    """Parse Taytsh source code into a TModule AST."""
    strict_math, strict_tostring = _extract_pragmas(source)
    tokens = tokenize(source)
    parser = Parser(tokens)
    module = parser.parse_program()
    module.strict_math = strict_math
    module.strict_tostring = strict_tostring
    return module


def check(source: str) -> list[CheckError]:
    """Parse and type-check Taytsh source. Returns list of errors (empty = ok)."""
    module = parse(source)
    return check_module(module)


def emit(module: TModule) -> str:
    """Emit a `TModule` AST to Taytsh textual syntax."""
    return to_source(module)
