"""Taytsh parser and typechecker — public API."""

from __future__ import annotations

from .ast import TModule
from .check import CheckError, check as check_module
from .emit import to_source
from .parse import ParseError, Parser
from .tokens import TokenizeError, tokenize


def parse(source: str) -> TModule:
    """Parse Taytsh source code into a TModule AST."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()


def check(source: str) -> list[CheckError]:
    """Parse and type-check Taytsh source. Returns list of errors (empty = ok)."""
    module = parse(source)
    return check_module(module)


def emit(module: TModule) -> str:
    """Emit a `TModule` AST to Taytsh textual syntax."""
    return to_source(module)
