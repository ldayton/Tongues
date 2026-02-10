"""Taytsh parser — public API."""

from __future__ import annotations

from .ast import TModule
from .parse import ParseError, Parser
from .tokens import TokenizeError, tokenize


def parse(source: str) -> TModule:
    """Parse Taytsh source code into a TModule AST."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()
