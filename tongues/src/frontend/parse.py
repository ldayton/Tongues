"""Phase 2 alternative: Parse Python source to dict-based AST (self-contained).

Written in the Tongues subset with no external dependencies.
"""

# Type aliases
ASTNode = dict[str, object]


class ParseError(Exception):
    """Parse error with location info."""

    def __init__(self, msg: str, lineno: int, col: int):
        self.msg: str = msg
        self.lineno: int = lineno
        self.col: int = col
        super().__init__(msg)


# Token types
TK_NAME = "NAME"
TK_NUMBER = "NUMBER"
TK_STRING = "STRING"
TK_NEWLINE = "NEWLINE"
TK_INDENT = "INDENT"
TK_DEDENT = "DEDENT"
TK_ENDMARKER = "ENDMARKER"
TK_OP = "OP"
TK_ERRORTOKEN = "ERRORTOKEN"

# Keywords (note: 'match' and 'case' are soft keywords, handled contextually)
KEYWORDS: set[str] = {
    "def",
    "class",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "try",
    "except",
    "finally",
    "return",
    "raise",
    "pass",
    "break",
    "continue",
    "import",
    "from",
    "as",
    "and",
    "or",
    "not",
    "in",
    "is",
    "None",
    "True",
    "False",
    "assert",
    "del",
    "global",
    "nonlocal",
    "lambda",
    "yield",
    "async",
    "await",
    "with",
}

# Multi-character operators (sorted by length descending for matching)
MULTI_OPS: list[str] = [
    "**=",
    "//=",
    ">>=",
    "<<=",
    "...",
    "->",
    "**",
    "//",
    "<<",
    ">>",
    "<=",
    ">=",
    "==",
    "!=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "@=",
    "&=",
    "|=",
    "^=",
    ":=",
]

# Single-character operators and delimiters
SINGLE_OPS: set[str] = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "@",
    "&",
    "|",
    "^",
    "~",
    "<",
    ">",
    "=",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ",",
    ":",
    ";",
    ".",
}


class Token:
    """A token with type, value, and position."""

    def __init__(self, type_: str, value: str, lineno: int, col: int):
        self.type: str = type_
        self.value: str = value
        self.lineno: int = lineno
        self.col: int = col

    def __repr__(self) -> str:
        return (
            "Token("
            + self.type
            + ", "
            + repr(self.value)
            + ", "
            + str(self.lineno)
            + ", "
            + str(self.col)
            + ")"
        )


def is_digit(c: str) -> bool:
    """Check if character is a digit."""
    return c >= "0" and c <= "9"


def is_alpha(c: str) -> bool:
    """Check if character is alphabetic or underscore."""
    return (c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or c == "_"


def is_alnum(c: str) -> bool:
    """Check if character is alphanumeric or underscore."""
    return is_alpha(c) or is_digit(c)


def is_whitespace(c: str) -> bool:
    """Check if character is whitespace (not newline)."""
    return c == " " or c == "\t"


def tokenize(source: str) -> list[Token]:
    """Tokenize Python source code."""
    tokens: list[Token] = []
    lines = source.split("\n")
    indent_stack: list[int] = [0]
    lineno = 1
    num_lines = len(lines)

    # Track if we're inside brackets (no INDENT/DEDENT inside)
    bracket_depth = 0
    bracket_stack: list[tuple[str, int]] = []

    while lineno <= num_lines:
        line = lines[lineno - 1]
        col = 0
        line_len = len(line)

        # Handle empty lines and comment-only lines
        # Skip leading whitespace to check for empty/comment
        temp_col = 0
        while temp_col < line_len and is_whitespace(line[temp_col]):
            temp_col += 1

        # Empty line or comment-only line - skip entirely (no tokens)
        if temp_col >= line_len or line[temp_col] == "#":
            lineno += 1
            continue

        # Handle indentation at start of non-blank line (only if not in brackets)
        if bracket_depth == 0:
            indent = 0
            while col < line_len and line[col] == " ":
                indent += 1
                col += 1
            while col < line_len and line[col] == "\t":
                indent += 8
                col += 1

            current_indent = indent_stack[len(indent_stack) - 1]
            if indent > current_indent:
                indent_stack.append(indent)
                tokens.append(Token(TK_INDENT, "", lineno, 0))
            elif indent < current_indent:
                while (
                    len(indent_stack) > 1
                    and indent_stack[len(indent_stack) - 1] > indent
                ):
                    indent_stack.pop()
                    tokens.append(Token(TK_DEDENT, "", lineno, 0))
                if indent_stack[len(indent_stack) - 1] != indent:
                    raise ParseError("inconsistent indentation", lineno, col)

        # Skip leading whitespace (already counted for indentation)
        while col < line_len and is_whitespace(line[col]):
            col += 1

        # Process tokens on this line
        while col < line_len:
            c = line[col]

            # Skip whitespace
            if is_whitespace(c):
                col += 1
                continue

            # Comment - rest of line
            if c == "#":
                break

            # Line continuation
            if c == "\\" and col + 1 >= line_len:
                # Skip the backslash and merge with next line
                lineno += 1
                if lineno <= num_lines:
                    line = line + lines[lineno - 1]
                    line_len = len(line)
                    col += 1
                continue

            # String literal
            if c == '"' or c == "'":
                tok, new_col, new_lineno, new_line = scan_string(
                    line, col, lineno, lines
                )
                tokens.append(tok)
                col = new_col
                if new_lineno != lineno:
                    lineno = new_lineno
                    line = new_line
                    line_len = len(line)
                continue

            # String prefix (r, b, f, u, rb, br, rf, fr)
            if c in "rRbBfFuU":
                prefix_len = 1
                if col + 1 < line_len and line[col + 1] in "rRbBfFuU":
                    prefix_len = 2
                if col + prefix_len < line_len and line[col + prefix_len] in "\"'":
                    tok, new_col, new_lineno, new_line = scan_string(
                        line, col, lineno, lines
                    )
                    tokens.append(tok)
                    col = new_col
                    if new_lineno != lineno:
                        lineno = new_lineno
                        line = new_line
                        line_len = len(line)
                    continue

            # Number
            if is_digit(c) or (
                c == "." and col + 1 < line_len and is_digit(line[col + 1])
            ):
                tok, new_col = scan_number(line, col, lineno)
                tokens.append(tok)
                col = new_col
                continue

            # Name or keyword
            if is_alpha(c):
                start = col
                while col < line_len and is_alnum(line[col]):
                    col += 1
                value = line[start:col]
                if value in KEYWORDS:
                    tokens.append(Token(value, value, lineno, start))
                else:
                    tokens.append(Token(TK_NAME, value, lineno, start))
                continue

            # Multi-character operators
            matched = False
            i = 0
            while i < len(MULTI_OPS):
                op = MULTI_OPS[i]
                op_len = len(op)
                if col + op_len <= line_len and line[col : col + op_len] == op:
                    tokens.append(Token(TK_OP, op, lineno, col))
                    col += op_len
                    if op == "(":
                        bracket_depth += 1
                    elif op == ")":
                        bracket_depth -= 1
                    matched = True
                    break
                i += 1
            if matched:
                continue

            # Single-character operators
            if c in SINGLE_OPS:
                tokens.append(Token(TK_OP, c, lineno, col))
                if c == "(" or c == "[" or c == "{":
                    bracket_depth += 1
                    bracket_stack.append((c, lineno))
                elif c == ")" or c == "]" or c == "}":
                    bracket_depth -= 1
                    if len(bracket_stack) > 0:
                        bracket_stack.pop()
                col += 1
                continue

            # Unknown character
            tokens.append(Token(TK_ERRORTOKEN, c, lineno, col))
            col += 1

        # End of line - emit NEWLINE if not in brackets
        if bracket_depth == 0:
            tokens.append(Token(TK_NEWLINE, "\n", lineno, line_len))

        lineno += 1

    # Check for unmatched brackets
    if len(bracket_stack) > 0:
        open_bracket = bracket_stack[0][0]
        open_line = bracket_stack[0][1]
        raise ParseError(
            "unmatched '" + open_bracket + "' at line " + str(open_line),
            open_line,
            0,
        )

    # Emit remaining DEDENTs
    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token(TK_DEDENT, "", lineno, 0))

    tokens.append(Token(TK_ENDMARKER, "", lineno, 0))
    return tokens


def scan_string(
    line: str, col: int, lineno: int, lines: list[str]
) -> tuple[Token, int, int, str]:
    """Scan a string literal, handling prefixes and multi-line strings."""
    start_col = col
    start_lineno = lineno

    # Skip prefix, detect raw
    is_raw = False
    while col < len(line) and line[col] in "rRbBfFuU":
        if line[col] in "rR":
            is_raw = True
        col += 1

    # Get quote character
    quote = line[col]
    col += 1

    # Check for triple quote
    triple = False
    if col + 1 < len(line) and line[col] == quote and line[col + 1] == quote:
        triple = True
        col += 2

    # Scan string content
    value_start = start_col
    current_line = line
    current_lineno = lineno

    while True:
        if triple:
            # Look for triple quote end
            while col < len(current_line):
                c = current_line[col]
                if c == "\\" and col + 1 < len(current_line) and not is_raw:
                    col += 2
                    continue
                if (
                    c == quote
                    and col + 2 < len(current_line)
                    and current_line[col + 1] == quote
                    and current_line[col + 2] == quote
                ):
                    col += 3
                    value = current_line[value_start:col]
                    return (
                        Token(TK_STRING, value, start_lineno, start_col),
                        col,
                        current_lineno,
                        current_line,
                    )
                col += 1
            # Move to next line for triple-quoted strings
            current_lineno += 1
            if current_lineno > len(lines):
                raise ParseError("unterminated string literal", start_lineno, start_col)
            current_line = current_line + "\n" + lines[current_lineno - 1]
        else:
            # Look for single quote end
            while col < len(current_line):
                c = current_line[col]
                if c == "\\" and not is_raw:
                    if col + 1 >= len(current_line):
                        # Line continuation in string literal
                        current_lineno += 1
                        if current_lineno > len(lines):
                            raise ParseError(
                                "unterminated string literal", start_lineno, start_col
                            )
                        current_line = current_line + "\n" + lines[current_lineno - 1]
                        col += 2
                        continue
                    col += 2
                    continue
                if c == quote:
                    col += 1
                    value = current_line[value_start:col]
                    return (
                        Token(TK_STRING, value, start_lineno, start_col),
                        col,
                        current_lineno,
                        current_line,
                    )
                if c == "\n":
                    raise ParseError("unterminated string literal", start_lineno, start_col)
                col += 1
            raise ParseError("unterminated string literal", start_lineno, start_col)


def scan_number(line: str, col: int, lineno: int) -> tuple[Token, int]:
    """Scan a numeric literal."""
    start = col
    line_len = len(line)

    # Check for hex, octal, binary
    if col + 1 < line_len and line[col] == "0":
        next_c = line[col + 1]
        if next_c in "xX":
            col += 2
            while col < line_len and (
                is_digit(line[col]) or line[col] in "abcdefABCDEF_"
            ):
                col += 1
            return Token(TK_NUMBER, line[start:col], lineno, start), col
        if next_c in "oO":
            col += 2
            while col < line_len and (
                line[col] >= "0" and line[col] <= "7" or line[col] == "_"
            ):
                col += 1
            return Token(TK_NUMBER, line[start:col], lineno, start), col
        if next_c in "bB":
            col += 2
            while col < line_len and line[col] in "01_":
                col += 1
            return Token(TK_NUMBER, line[start:col], lineno, start), col

    # Integer or float
    while col < line_len and (is_digit(line[col]) or line[col] == "_"):
        col += 1

    # Decimal part
    if col < line_len and line[col] == ".":
        col += 1
        while col < line_len and (is_digit(line[col]) or line[col] == "_"):
            col += 1

    # Exponent
    if col < line_len and line[col] in "eE":
        col += 1
        if col < line_len and line[col] in "+-":
            col += 1
        while col < line_len and (is_digit(line[col]) or line[col] == "_"):
            col += 1

    # Complex suffix
    if col < line_len and line[col] in "jJ":
        col += 1

    return Token(TK_NUMBER, line[start:col], lineno, start), col


class Parser:
    """Recursive descent parser for Python."""

    def __init__(self, tokens: list[Token]):
        self.tokens: list[Token] = tokens
        self.pos: int = 0
        self.func_depth: int = 0
        self.loop_depth: int = 0
        self.async_depth: int = 0
        self.func_has_yield: bool = False
        self.class_depth: int = 0
        self.comp_depth: int = 0
        self.comp_iter_depth: int = 0
        self.comp_target_names: list[set[str]] = []
        self.in_class_comp: bool = False

    def current(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[len(self.tokens) - 1]

    def peek(self, offset: int) -> Token:
        """Peek at token at offset from current."""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[len(self.tokens) - 1]

    def advance(self) -> Token:
        """Consume and return current token."""
        tok = self.current()
        if self.pos < len(self.tokens):
            self.pos += 1
        return tok

    def match(self, type_or_value: str) -> bool:
        """Check if current token matches type or value."""
        tok = self.current()
        if tok.type == type_or_value:
            return True
        # Only match by value for keywords (not special token types)
        if type_or_value in (
            TK_NAME,
            TK_NUMBER,
            TK_STRING,
            TK_NEWLINE,
            TK_INDENT,
            TK_DEDENT,
            TK_ENDMARKER,
            TK_OP,
        ):
            return False
        return tok.value == type_or_value

    def match_op(self, value: str) -> bool:
        """Check if current token is operator with given value."""
        tok = self.current()
        return tok.type == TK_OP and tok.value == value

    def expect(self, type_or_value: str) -> Token:
        """Consume token matching type or value, or raise error."""
        tok = self.current()
        if tok.type == type_or_value:
            return self.advance()
        # Only match by value for keywords (not special token types)
        if type_or_value not in (
            TK_NAME,
            TK_NUMBER,
            TK_STRING,
            TK_NEWLINE,
            TK_INDENT,
            TK_DEDENT,
            TK_ENDMARKER,
            TK_OP,
        ):
            if tok.value == type_or_value:
                return self.advance()
        raise ParseError(
            "invalid syntax at line "
            + str(tok.lineno)
            + ", column "
            + str(tok.col),
            tok.lineno,
            tok.col,
        )

    def prev_token(self) -> Token:
        """Get the previously consumed token."""
        if self.pos > 0:
            return self.tokens[self.pos - 1]
        return self.tokens[0]

    def expect_op(self, value: str) -> Token:
        """Consume operator token with given value."""
        tok = self.current()
        if tok.type == TK_OP and tok.value == value:
            return self.advance()
        raise ParseError(
            "invalid syntax at line "
            + str(tok.lineno)
            + ", column "
            + str(tok.col),
            tok.lineno,
            tok.col,
        )

    def skip_newlines(self) -> None:
        """Skip NEWLINE tokens."""
        while self.match(TK_NEWLINE):
            self.advance()

    def error(self, msg: str) -> ParseError:
        """Create parse error at current position."""
        tok = self.current()
        return ParseError(msg, tok.lineno, tok.col)

    def check_walrus_scope(self, target: ASTNode) -> None:
        """Check walrus operator scope restrictions in comprehensions."""
        if self.comp_iter_depth > 0:
            raise self.error("assignment expression cannot be used in a comprehension iterable expression")
        if self.in_class_comp:
            raise self.error("assignment expression within a comprehension cannot be used in a class body")
        if self.comp_depth > 0 and len(self.comp_target_names) > 0:
            name = target.get("id") if isinstance(target, dict) else None
            if isinstance(name, str):
                i = 0
                while i < len(self.comp_target_names):
                    if name in self.comp_target_names[i]:
                        raise self.error(
                            "assignment expression cannot rebind comprehension iteration variable '"
                            + name
                            + "'"
                        )
                    i += 1

    def is_match_statement(self) -> bool:
        """Check if current 'match' token starts a match statement (soft keyword check)."""
        # A match statement is: match <expr> :
        # We need to look ahead to find the colon that ends the subject expression
        # This is a heuristic - we look for a ':' followed by NEWLINE at the same bracket depth
        pos = self.pos + 1
        depth = 0
        while pos < len(self.tokens):
            tok = self.tokens[pos]
            if tok.type == TK_OP:
                if tok.value in "([{":
                    depth += 1
                elif tok.value in ")]}":
                    depth -= 1
                elif tok.value == ":" and depth == 0:
                    # Check if next is NEWLINE (match statement) or more expr (dict/slice)
                    if pos + 1 < len(self.tokens):
                        next_tok = self.tokens[pos + 1]
                        if next_tok.type == TK_NEWLINE:
                            return True
                    return False
            elif tok.type == TK_NEWLINE and depth == 0:
                return False
            pos += 1
        return False

    def is_type_statement(self) -> bool:
        """Check if current 'type' token starts a type alias statement (soft keyword check)."""
        pos = self.pos + 1
        if pos >= len(self.tokens):
            return False
        if self.tokens[pos].type != TK_NAME:
            return False
        pos += 1
        depth = 0
        while pos < len(self.tokens):
            tok = self.tokens[pos]
            if tok.type == TK_OP:
                if tok.value in "([{":
                    depth += 1
                elif tok.value in ")]}":
                    depth -= 1
                elif depth == 0 and tok.value == "=":
                    return True
            elif tok.type == TK_NEWLINE and depth == 0:
                return False
            pos += 1
        return False

    def parse_type_alias_stmt(self) -> ASTNode:
        """Parse type alias statement: type Name[params] = value."""
        tok = self.advance()  # consume 'type' soft keyword
        name_tok = self.expect(TK_NAME)
        name_node = make_node(
            "Name",
            name_tok.lineno,
            name_tok.col,
            {"id": name_tok.value, "ctx": {"_type": "Store"}},
        )
        end_from_token(name_node, name_tok)
        type_params = self.parse_type_params()
        self.expect_op("=")
        value = self.parse_test()
        node = make_node(
            "TypeAlias",
            tok.lineno,
            tok.col,
            {"name": name_node, "type_params": type_params, "value": value},
        )
        end_from_node(node, value)
        return node

    # --- Module parsing ---

    def parse_module(self) -> ASTNode:
        """Parse a module (file_input)."""
        tok = self.current()
        body: list[ASTNode] = []
        self.skip_newlines()
        while not self.match(TK_ENDMARKER):
            stmt = self.parse_stmt()
            if stmt is not None:
                if isinstance(stmt, list):
                    i = 0
                    while i < len(stmt):
                        body.append(stmt[i])
                        i += 1
                else:
                    body.append(stmt)
            self.skip_newlines()
        node = make_node(
            "Module", tok.lineno, tok.col, {"body": body, "type_ignores": []}
        )
        if len(body) > 0:
            end_from_node(node, body[len(body) - 1])
        return node

    # --- Statement parsing ---

    def parse_stmt(self) -> ASTNode | list[ASTNode] | None:
        """Parse a statement."""
        self.skip_newlines()
        tok = self.current()

        # Compound statements
        if tok.value == "def":
            return self.parse_funcdef()
        if tok.value == "async":
            return self.parse_async_stmt()
        if tok.value == "class":
            return self.parse_classdef()
        if tok.value == "if":
            return self.parse_if_stmt()
        if tok.value == "while":
            return self.parse_while_stmt()
        if tok.value == "for":
            return self.parse_for_stmt()
        if tok.value == "try":
            return self.parse_try_stmt()
        if tok.value == "with":
            return self.parse_with_stmt()
        # 'match' is a soft keyword - only a match statement if followed by expr and ':'
        if tok.type == TK_NAME and tok.value == "match":
            if self.is_match_statement():
                return self.parse_match_stmt()
        # 'type' is a soft keyword - only a type alias if followed by NAME then '=' or '['
        if tok.type == TK_NAME and tok.value == "type":
            if self.is_type_statement():
                return self.parse_type_alias_stmt()
        if tok.type == TK_OP and tok.value == "@":
            return self.parse_decorated()

        # Simple statements
        return self.parse_simple_stmt()

    def parse_simple_stmt(self) -> ASTNode | list[ASTNode] | None:
        """Parse simple statement(s) on one line."""
        stmts: list[ASTNode] = []
        while True:
            stmt = self.parse_small_stmt()
            if stmt is not None:
                stmts.append(stmt)
            if self.match_op(";"):
                self.advance()
                if self.match(TK_NEWLINE) or self.match(TK_ENDMARKER):
                    break
            else:
                break
        if self.match(TK_NEWLINE):
            self.advance()
        if len(stmts) == 1:
            return stmts[0]
        return stmts

    def parse_small_stmt(self) -> ASTNode | None:
        """Parse a single simple statement."""
        tok = self.current()

        if tok.value == "return":
            return self.parse_return_stmt()
        if tok.value == "raise":
            return self.parse_raise_stmt()
        if tok.value == "pass":
            self.advance()
            return end_from_token(make_node("Pass", tok.lineno, tok.col), tok)
        if tok.value == "break":
            if self.loop_depth == 0:
                raise self.error("'break' outside loop")
            self.advance()
            return end_from_token(make_node("Break", tok.lineno, tok.col), tok)
        if tok.value == "continue":
            if self.loop_depth == 0:
                raise self.error("'continue' outside loop")
            self.advance()
            return end_from_token(make_node("Continue", tok.lineno, tok.col), tok)
        if tok.value == "import":
            return self.parse_import_stmt()
        if tok.value == "from":
            return self.parse_from_import_stmt()
        if tok.value == "assert":
            return self.parse_assert_stmt()
        if tok.value == "del":
            return self.parse_del_stmt()
        if tok.value == "global":
            return self.parse_global_stmt()
        if tok.value == "nonlocal":
            return self.parse_nonlocal_stmt()
        if tok.value == "yield":
            return self.parse_yield_stmt()

        # Expression statement (possibly assignment)
        return self.parse_expr_stmt()

    def parse_return_stmt(self) -> ASTNode:
        """Parse return statement."""
        if self.func_depth == 0:
            raise self.error("'return' outside function")
        tok = self.expect("return")
        value: ASTNode | None = None
        if (
            not self.match(TK_NEWLINE)
            and not self.match_op(";")
            and not self.match(TK_ENDMARKER)
        ):
            value = self.parse_testlist_star_expr()
            if isinstance(value, dict) and value.get("_type") == "Starred":
                raise self.error("starred expression is not allowed here")
        node = make_node("Return", tok.lineno, tok.col, {"value": value})
        if value is not None:
            end_from_node(node, value)
        else:
            end_from_token(node, tok)
        return node

    def parse_raise_stmt(self) -> ASTNode:
        """Parse raise statement."""
        tok = self.expect("raise")
        exc: ASTNode | None = None
        cause: ASTNode | None = None
        if (
            not self.match(TK_NEWLINE)
            and not self.match_op(";")
            and not self.match(TK_ENDMARKER)
        ):
            exc = self.parse_test()
            if self.match("from"):
                self.advance()
                cause = self.parse_test()
        node = make_node("Raise", tok.lineno, tok.col, {"exc": exc, "cause": cause})
        if cause is not None:
            end_from_node(node, cause)
        elif exc is not None:
            end_from_node(node, exc)
        else:
            end_from_token(node, tok)
        return node

    def parse_import_stmt(self) -> ASTNode:
        """Parse import statement."""
        tok = self.expect("import")
        names = self.parse_dotted_as_names()
        node = make_node("Import", tok.lineno, tok.col, {"names": names})
        end_from_token(node, self.prev_token())
        return node

    def parse_from_import_stmt(self) -> ASTNode:
        """Parse from ... import statement."""
        tok = self.expect("from")
        level = 0
        while self.match_op(".") or self.match_op("..."):
            if self.current().value == "...":
                level += 3
            else:
                level += 1
            self.advance()

        module: str | None = None
        if self.match(TK_NAME):
            module = self.parse_dotted_name()

        self.expect("import")

        names: list[ASTNode] = []
        if self.match_op("*"):
            if self.func_depth > 0:
                raise self.error("import * only allowed at module level")
            self.advance()
            names.append({"_type": "alias", "name": "*", "asname": None})
        elif self.match_op("("):
            self.advance()
            names = self.parse_import_as_names()
            self.expect_op(")")
        else:
            names = self.parse_import_as_names()
            if self.prev_token().type == TK_OP and self.prev_token().value == ",":
                raise self.error("trailing comma not allowed without parentheses")

        if module == "__future__":
            _FUTURE_FEATURES: set[str] = {
                "nested_scopes",
                "generators",
                "division",
                "absolute_import",
                "with_statement",
                "print_function",
                "unicode_literals",
                "barry_as_FLUFL",
                "generator_stop",
                "annotations",
            }
            i = 0
            while i < len(names):
                nm = names[i]
                if isinstance(nm, dict):
                    fname = nm.get("name")
                    if isinstance(fname, str):
                        if fname == "braces":
                            raise self.error("not a chance")
                        if fname not in _FUTURE_FEATURES:
                            raise self.error(
                                "future feature " + fname + " is not defined"
                            )
                i += 1

        node = make_node(
            "ImportFrom",
            tok.lineno,
            tok.col,
            {"module": module, "names": names, "level": level},
        )
        end_from_token(node, self.prev_token())
        return node

    def parse_dotted_name(self) -> str:
        """Parse dotted name like a.b.c."""
        parts: list[str] = []
        tok = self.expect(TK_NAME)
        parts.append(tok.value)
        while self.match_op("."):
            self.advance()
            tok = self.expect(TK_NAME)
            parts.append(tok.value)
        return ".".join(parts)

    def parse_dotted_as_names(self) -> list[ASTNode]:
        """Parse import names: a.b as c, d.e as f."""
        names: list[ASTNode] = []
        names.append(self.parse_dotted_as_name())
        while self.match_op(","):
            self.advance()
            names.append(self.parse_dotted_as_name())
        return names

    def parse_dotted_as_name(self) -> ASTNode:
        """Parse a.b.c as d."""
        name = self.parse_dotted_name()
        asname: str | None = None
        if self.match("as"):
            self.advance()
            asname = self.expect(TK_NAME).value
        return {"_type": "alias", "name": name, "asname": asname}

    def parse_import_as_names(self) -> list[ASTNode]:
        """Parse import names: a as b, c as d."""
        names: list[ASTNode] = []
        names.append(self.parse_import_as_name())
        while self.match_op(","):
            self.advance()
            if self.match_op(")") or self.match(TK_NEWLINE):
                break
            names.append(self.parse_import_as_name())
        return names

    def parse_import_as_name(self) -> ASTNode:
        """Parse name as alias."""
        tok = self.expect(TK_NAME)
        name = tok.value
        asname: str | None = None
        if self.match("as"):
            self.advance()
            asname = self.expect(TK_NAME).value
        return {"_type": "alias", "name": name, "asname": asname}

    def parse_assert_stmt(self) -> ASTNode:
        """Parse assert statement."""
        tok = self.expect("assert")
        test = self.parse_test()
        msg: ASTNode | None = None
        if self.match_op(","):
            self.advance()
            msg = self.parse_test()
        node = make_node("Assert", tok.lineno, tok.col, {"test": test, "msg": msg})
        if msg is not None:
            end_from_node(node, msg)
        else:
            end_from_node(node, test)
        return node

    def parse_del_stmt(self) -> ASTNode:
        """Parse del statement."""
        tok = self.expect("del")
        targets = self.parse_exprlist()
        set_context_list(targets, "Del")
        node = make_node("Delete", tok.lineno, tok.col, {"targets": targets})
        if len(targets) > 0:
            end_from_node(node, targets[len(targets) - 1])
        else:
            end_from_token(node, tok)
        return node

    def parse_global_stmt(self) -> ASTNode:
        """Parse global statement."""
        tok = self.expect("global")
        names: list[str] = []
        names.append(self.expect(TK_NAME).value)
        while self.match_op(","):
            self.advance()
            names.append(self.expect(TK_NAME).value)
        node = make_node("Global", tok.lineno, tok.col, {"names": names})
        end_from_token(node, self.prev_token())
        return node

    def parse_nonlocal_stmt(self) -> ASTNode:
        """Parse nonlocal statement."""
        if self.func_depth == 0:
            raise self.error("nonlocal declaration not allowed at module level")
        tok = self.expect("nonlocal")
        names: list[str] = []
        names.append(self.expect(TK_NAME).value)
        while self.match_op(","):
            self.advance()
            names.append(self.expect(TK_NAME).value)
        node = make_node("Nonlocal", tok.lineno, tok.col, {"names": names})
        end_from_token(node, self.prev_token())
        return node

    def parse_yield_stmt(self) -> ASTNode:
        """Parse yield statement as Expr(Yield(...))."""
        tok = self.current()
        yield_expr = self.parse_yield_expr()
        return end_from_node(
            make_node("Expr", tok.lineno, tok.col, {"value": yield_expr}), yield_expr
        )

    def parse_yield_expr(self) -> ASTNode:
        """Parse yield expression."""
        if self.func_depth == 0:
            raise self.error("'yield' outside function")
        tok = self.expect("yield")
        if self.match("from"):
            if self.async_depth > 0:
                raise self.error("'yield from' inside async function")
            self.advance()
            value = self.parse_test()
            self.func_has_yield = True
            return end_from_node(
                make_node("YieldFrom", tok.lineno, tok.col, {"value": value}), value
            )
        value: ASTNode | None = None
        if (
            not self.match(TK_NEWLINE)
            and not self.match_op(";")
            and not self.match_op(")")
            and not self.match(TK_ENDMARKER)
        ):
            value = self.parse_testlist_star_expr()
            if isinstance(value, dict) and value.get("_type") == "Starred":
                raise self.error("starred expression is not allowed here")
        self.func_has_yield = True
        node = make_node("Yield", tok.lineno, tok.col, {"value": value})
        if value is not None:
            end_from_node(node, value)
        else:
            end_from_token(node, tok)
        return node

    def parse_expr_stmt(self) -> ASTNode | None:
        """Parse expression statement (may be assignment)."""
        tok = self.current()
        if self.match(TK_NEWLINE) or self.match(TK_ENDMARKER):
            return None

        # Parse first expression
        target = self.parse_testlist_star_expr()

        # Check for walrus operator
        if self.match_op(":="):
            self.advance()
            value = self.parse_test()
            validate_target(target, "Store", False, True, False)
            self.check_walrus_scope(target)
            if "ctx" in target:
                target["ctx"] = {"_type": "Store"}
            return end_from_node(
                make_node(
                    "NamedExpr",
                    tok.lineno,
                    tok.col,
                    {"target": target, "value": value},
                ),
                value,
            )

        # Check for annotated assignment
        if self.match_op(":"):
            self.advance()
            annotation = self.parse_test()
            value: ASTNode | None = None
            if self.match_op("="):
                self.advance()
                value = self.parse_testlist_star_expr()
                if isinstance(value, dict) and value.get("_type") == "Starred":
                    raise self.error("starred expression is not allowed here")
            validate_target(target, "Store", False, False, True)
            if "ctx" in target:
                target["ctx"] = {"_type": "Store"}
            simple = 1
            if target.get("_type") != "Name":
                simple = 0
            node = make_node(
                "AnnAssign",
                tok.lineno,
                tok.col,
                {
                    "target": target,
                    "annotation": annotation,
                    "value": value,
                    "simple": simple,
                },
            )
            if value is not None:
                end_from_node(node, value)
            else:
                end_from_node(node, annotation)
            return node

        # Check for augmented assignment
        aug_ops = [
            "+=",
            "-=",
            "*=",
            "/=",
            "//=",
            "%=",
            "**=",
            "@=",
            "&=",
            "|=",
            "^=",
            ">>=",
            "<<=",
        ]
        i = 0
        while i < len(aug_ops):
            if self.match_op(aug_ops[i]):
                op_tok = self.advance()
                value = self.parse_testlist_star_expr()
                validate_target(target, "Store", True, False, False)
                if "ctx" in target:
                    target["ctx"] = {"_type": "Store"}
                op = augassign_op(op_tok.value)
                return end_from_node(
                    make_node(
                        "AugAssign",
                        tok.lineno,
                        tok.col,
                        {"target": target, "op": op, "value": value},
                    ),
                    value,
                )
            i += 1

        # Check for regular assignment
        if self.match_op("="):
            targets: list[ASTNode] = [target]
            while self.match_op("="):
                self.advance()
                next_expr = self.parse_testlist_star_expr()
                targets.append(next_expr)
            # Last one is the value
            value = targets.pop()
            # Validate starred in value (RHS)
            if isinstance(value, dict) and value.get("_type") == "Starred":
                raise self.error("starred expression is not allowed here")
            # Validate starred in targets
            j = 0
            while j < len(targets):
                t = targets[j]
                if isinstance(t, dict):
                    tt = t.get("_type")
                    if tt == "Starred":
                        raise self.error("starred assignment target must be in a list or tuple")
                    if tt in ("Tuple", "List"):
                        telts = t.get("elts")
                        if isinstance(telts, list):
                            star_count = 0
                            si = 0
                            while si < len(telts):
                                if isinstance(telts[si], dict) and telts[si].get("_type") == "Starred":
                                    star_count += 1
                                si += 1
                            if star_count > 1:
                                raise self.error("multiple starred expressions in assignment")
                set_context(targets[j], "Store")
                j += 1
            return end_from_node(
                make_node(
                    "Assign", tok.lineno, tok.col, {"targets": targets, "value": value}
                ),
                value,
            )

        # Just an expression
        if isinstance(target, dict) and target.get("_type") == "Starred":
            raise self.error("starred expression is not allowed here")
        return end_from_node(
            make_node("Expr", tok.lineno, tok.col, {"value": target}), target
        )

    # --- Compound statements ---

    def parse_type_params(self) -> list[ASTNode]:
        """Parse PEP 695 type parameter list: [T, U, *Ts, **P]."""
        if not self.match_op("["):
            return []
        self.advance()
        params: list[ASTNode] = []
        while not self.match_op("]"):
            if self.match_op(","):
                self.advance()
                continue
            tok = self.current()
            if self.match_op("**"):
                self.advance()
                name_tok = self.expect(TK_NAME)
                node = make_node(
                    "ParamSpec",
                    tok.lineno,
                    tok.col,
                    {"name": name_tok.value, "default_value": None},
                )
                end_from_token(node, name_tok)
            elif self.match_op("*"):
                self.advance()
                name_tok = self.expect(TK_NAME)
                node = make_node(
                    "TypeVarTuple",
                    tok.lineno,
                    tok.col,
                    {"name": name_tok.value, "default_value": None},
                )
                end_from_token(node, name_tok)
            else:
                name_tok = self.expect(TK_NAME)
                bound: ASTNode | None = None
                if self.match_op(":"):
                    self.advance()
                    bound = self.parse_test()
                node = make_node(
                    "TypeVar",
                    tok.lineno,
                    tok.col,
                    {"name": name_tok.value, "bound": bound, "default_value": None},
                )
                end_from_token(node, name_tok)
            if self.match_op("="):
                self.advance()
                default = self.parse_test()
                node["default_value"] = default
                end_from_node(node, default)
            params.append(node)
        self.expect_op("]")
        return params

    def parse_funcdef(self, is_async: bool = False) -> ASTNode:
        """Parse function definition."""
        tok = self.expect("def")
        name = self.expect(TK_NAME).value
        type_params = self.parse_type_params()
        params = self.parse_parameters()
        returns: ASTNode | None = None
        if self.match_op("->"):
            self.advance()
            returns = self.parse_test()
        self.expect_op(":")
        saved_func_depth = self.func_depth
        saved_loop_depth = self.loop_depth
        saved_async_depth = self.async_depth
        saved_has_yield = self.func_has_yield
        self.func_depth = self.func_depth + 1
        self.loop_depth = 0
        self.async_depth = 1 if is_async else 0
        self.func_has_yield = False
        body = self.parse_suite()
        func_had_yield = self.func_has_yield
        if is_async and func_had_yield:
            _check_async_generator_return(body, tok)
        self.func_depth = saved_func_depth
        self.loop_depth = saved_loop_depth
        self.async_depth = saved_async_depth
        self.func_has_yield = saved_has_yield
        node = make_node(
            "FunctionDef",
            tok.lineno,
            tok.col,
            {
                "name": name,
                "args": params,
                "body": body,
                "decorator_list": [],
                "returns": returns,
                "type_params": type_params,
            },
        )
        if len(body) > 0:
            end_from_node(node, body[len(body) - 1])
        return node

    def parse_async_stmt(self) -> ASTNode:
        """Parse async statement (async def, async for, async with)."""
        tok = self.expect("async")
        if self.match("def"):
            func = self.parse_funcdef(is_async=True)
            func["_type"] = "AsyncFunctionDef"
            func["lineno"] = tok.lineno
            func["col_offset"] = tok.col
            return func
        if self.match("for"):
            for_stmt = self.parse_for_stmt()
            for_stmt["_type"] = "AsyncFor"
            for_stmt["lineno"] = tok.lineno
            for_stmt["col_offset"] = tok.col
            return for_stmt
        if self.match("with"):
            with_stmt = self.parse_with_stmt()
            with_stmt["_type"] = "AsyncWith"
            with_stmt["lineno"] = tok.lineno
            with_stmt["col_offset"] = tok.col
            return with_stmt
        raise self.error("expected 'def', 'for', or 'with' after 'async'")

    def parse_parameters(self) -> ASTNode:
        """Parse function parameters."""
        self.expect_op("(")
        if self.match_op(")"):
            self.advance()
            return make_arguments()
        params = self.parse_typedargslist()
        self.expect_op(")")
        return params

    def parse_typedargslist(self) -> ASTNode:
        """Parse typed argument list."""
        args: list[ASTNode] = []
        posonlyargs: list[ASTNode] = []
        kwonlyargs: list[ASTNode] = []
        defaults: list[ASTNode] = []
        kw_defaults: list[ASTNode | None] = []
        vararg: ASTNode | None = None
        kwarg: ASTNode | None = None
        in_kwonly = False
        has_default = False
        seen_names: set[str] = set()
        bare_star = False

        while not self.match_op(")"):
            if self.match_op(","):
                self.advance()
                continue

            # Positional-only marker
            if self.match_op("/"):
                self.advance()
                posonlyargs = args[:]
                args = []
                has_default = False
                continue

            # *args or bare *
            if self.match_op("*"):
                self.advance()
                in_kwonly = True
                if self.match(TK_NAME):
                    vararg = self.parse_arg()
                    pname = vararg.get("arg")
                    if isinstance(pname, str):
                        if pname in seen_names:
                            raise self.error("duplicate argument '" + pname + "'")
                        seen_names.add(pname)
                    bare_star = False
                else:
                    bare_star = True
                continue

            # **kwargs
            if self.match_op("**"):
                self.advance()
                kwarg = self.parse_arg()
                pname = kwarg.get("arg")
                if isinstance(pname, str):
                    if pname in seen_names:
                        raise self.error("duplicate argument '" + pname + "'")
                    seen_names.add(pname)
                continue

            # Regular argument
            arg = self.parse_arg()
            pname = arg.get("arg")
            if isinstance(pname, str):
                if pname in seen_names:
                    raise self.error("duplicate argument '" + pname + "'")
                seen_names.add(pname)
            default: ASTNode | None = None
            if self.match_op("="):
                self.advance()
                default = self.parse_test()

            if in_kwonly:
                kwonlyargs.append(arg)
                kw_defaults.append(default)
            else:
                if default is not None:
                    has_default = True
                elif has_default:
                    raise self.error("non-default argument follows default argument")
                args.append(arg)
                if default is not None:
                    defaults.append(default)

        if bare_star and len(kwonlyargs) == 0:
            raise self.error("named arguments must follow bare *")

        return {
            "_type": "arguments",
            "posonlyargs": posonlyargs,
            "args": args,
            "vararg": vararg,
            "kwonlyargs": kwonlyargs,
            "kw_defaults": kw_defaults,
            "kwarg": kwarg,
            "defaults": defaults,
        }

    def parse_arg(self) -> ASTNode:
        """Parse a single argument with optional annotation."""
        tok = self.expect(TK_NAME)
        name = tok.value
        annotation: ASTNode | None = None
        if self.match_op(":"):
            self.advance()
            annotation = self.parse_test()
        return {
            "_type": "arg",
            "arg": name,
            "annotation": annotation,
            "lineno": tok.lineno,
            "col_offset": tok.col,
            "end_lineno": tok.lineno,
            "end_col_offset": tok.col + len(name),
        }

    def parse_classdef(self) -> ASTNode:
        """Parse class definition."""
        tok = self.expect("class")
        name = self.expect(TK_NAME).value
        type_params = self.parse_type_params()
        bases: list[ASTNode] = []
        keywords: list[ASTNode] = []
        if self.match_op("("):
            self.advance()
            if not self.match_op(")"):
                bases, keywords = self.parse_arglist()
            self.expect_op(")")
            i = 0
            while i < len(bases):
                b = bases[i]
                if isinstance(b, dict) and b.get("_type") == "GeneratorExp":
                    raise self.error("cannot use generator expression in class bases")
                i += 1
        self.expect_op(":")
        self.class_depth += 1
        body = self.parse_suite()
        self.class_depth -= 1
        node = make_node(
            "ClassDef",
            tok.lineno,
            tok.col,
            {
                "name": name,
                "bases": bases,
                "keywords": keywords,
                "body": body,
                "decorator_list": [],
                "type_params": type_params,
            },
        )
        if len(body) > 0:
            end_from_node(node, body[len(body) - 1])
        return node

    def parse_decorated(self) -> ASTNode:
        """Parse decorated function or class."""
        decorators: list[ASTNode] = []
        while self.match_op("@"):
            tok = self.advance()
            decorator = self.parse_namedexpr_test()
            decorators.append(decorator)
            self.skip_newlines()

        if self.match("def"):
            func = self.parse_funcdef()
            func["decorator_list"] = decorators
            if decorators:
                func["lineno"] = decorators[0].get("lineno", func.get("lineno"))
                func["col_offset"] = decorators[0].get(
                    "col_offset", func.get("col_offset")
                )
            return func
        if self.match("async"):
            func = self.parse_async_stmt()
            func["decorator_list"] = decorators
            if decorators:
                func["lineno"] = decorators[0].get("lineno", func.get("lineno"))
                func["col_offset"] = decorators[0].get(
                    "col_offset", func.get("col_offset")
                )
            return func
        if self.match("class"):
            cls = self.parse_classdef()
            cls["decorator_list"] = decorators
            if decorators:
                cls["lineno"] = decorators[0].get("lineno", cls.get("lineno"))
                cls["col_offset"] = decorators[0].get(
                    "col_offset", cls.get("col_offset")
                )
            return cls
        raise self.error("expected 'def', 'async', or 'class' after decorator")

    def parse_if_stmt(self) -> ASTNode:
        """Parse if statement."""
        tok = self.expect("if")
        test = self.parse_namedexpr_test()
        self.expect_op(":")
        body = self.parse_suite()
        orelse: list[ASTNode] = []

        # Handle elif/else
        self.skip_newlines()
        if self.match("elif"):
            elif_stmt = self.parse_elif()
            orelse = [elif_stmt]
        elif self.match("else"):
            self.advance()
            self.expect_op(":")
            orelse = self.parse_suite()

        node = make_node(
            "If", tok.lineno, tok.col, {"test": test, "body": body, "orelse": orelse}
        )
        last = orelse if len(orelse) > 0 else body
        if len(last) > 0:
            end_from_node(node, last[len(last) - 1])
        return node

    def parse_elif(self) -> ASTNode:
        """Parse elif as nested If."""
        tok = self.expect("elif")
        test = self.parse_namedexpr_test()
        self.expect_op(":")
        body = self.parse_suite()
        orelse: list[ASTNode] = []

        self.skip_newlines()
        if self.match("elif"):
            elif_stmt = self.parse_elif()
            orelse = [elif_stmt]
        elif self.match("else"):
            self.advance()
            self.expect_op(":")
            orelse = self.parse_suite()

        node = make_node(
            "If", tok.lineno, tok.col, {"test": test, "body": body, "orelse": orelse}
        )
        last = orelse if len(orelse) > 0 else body
        if len(last) > 0:
            end_from_node(node, last[len(last) - 1])
        return node

    def parse_while_stmt(self) -> ASTNode:
        """Parse while statement."""
        tok = self.expect("while")
        test = self.parse_namedexpr_test()
        self.expect_op(":")
        self.loop_depth += 1
        body = self.parse_suite()
        self.loop_depth -= 1
        orelse: list[ASTNode] = []

        self.skip_newlines()
        if self.match("else"):
            self.advance()
            self.expect_op(":")
            orelse = self.parse_suite()

        node = make_node(
            "While", tok.lineno, tok.col, {"test": test, "body": body, "orelse": orelse}
        )
        last = orelse if len(orelse) > 0 else body
        if len(last) > 0:
            end_from_node(node, last[len(last) - 1])
        return node

    def parse_for_stmt(self) -> ASTNode:
        """Parse for statement."""
        tok = self.expect("for")
        target = self.parse_target_list()
        if isinstance(target, dict) and target.get("_type") == "Starred":
            raise self.error("starred assignment target must be in a list or tuple")
        set_context(target, "Store")
        self.expect("in")
        iter_expr = self.parse_testlist_star_expr()
        self.expect_op(":")
        self.loop_depth += 1
        body = self.parse_suite()
        self.loop_depth -= 1
        orelse: list[ASTNode] = []

        self.skip_newlines()
        if self.match("else"):
            self.advance()
            self.expect_op(":")
            orelse = self.parse_suite()

        node = make_node(
            "For",
            tok.lineno,
            tok.col,
            {"target": target, "iter": iter_expr, "body": body, "orelse": orelse},
        )
        last = orelse if len(orelse) > 0 else body
        if len(last) > 0:
            end_from_node(node, last[len(last) - 1])
        return node

    def parse_try_stmt(self) -> ASTNode:
        """Parse try statement."""
        tok = self.expect("try")
        self.expect_op(":")
        body = self.parse_suite()
        handlers: list[ASTNode] = []
        orelse: list[ASTNode] = []
        finalbody: list[ASTNode] = []
        is_star = False

        self.skip_newlines()

        # Parse except clauses
        while self.match("except"):
            handler_tok = self.advance()

            # Check for except*
            if self.match_op("*"):
                self.advance()
                is_star = True

            exc_type: ASTNode | None = None
            exc_name: str | None = None

            if not self.match_op(":"):
                exc_type = self.parse_test()
                if self.match("as"):
                    self.advance()
                    exc_name = self.expect(TK_NAME).value

            self.expect_op(":")
            handler_body = self.parse_suite()
            handler_node = make_node(
                "ExceptHandler",
                handler_tok.lineno,
                handler_tok.col,
                {"type": exc_type, "name": exc_name, "body": handler_body},
            )
            if len(handler_body) > 0:
                end_from_node(handler_node, handler_body[len(handler_body) - 1])
            handlers.append(handler_node)
            self.skip_newlines()

        # Parse else
        if self.match("else"):
            self.advance()
            self.expect_op(":")
            orelse = self.parse_suite()
            self.skip_newlines()

        # Parse finally
        if self.match("finally"):
            self.advance()
            self.expect_op(":")
            finalbody = self.parse_suite()

        type_name = "TryStar" if is_star else "Try"
        node = make_node(
            type_name,
            tok.lineno,
            tok.col,
            {
                "body": body,
                "handlers": handlers,
                "orelse": orelse,
                "finalbody": finalbody,
            },
        )
        if len(finalbody) > 0:
            end_from_node(node, finalbody[len(finalbody) - 1])
        elif len(orelse) > 0:
            end_from_node(node, orelse[len(orelse) - 1])
        elif len(handlers) > 0:
            end_from_node(node, handlers[len(handlers) - 1])
        elif len(body) > 0:
            end_from_node(node, body[len(body) - 1])
        return node

    def parse_with_stmt(self) -> ASTNode:
        """Parse with statement."""
        tok = self.expect("with")
        items: list[ASTNode] = []

        # First item
        items.append(self.parse_with_item())
        while self.match_op(","):
            self.advance()
            items.append(self.parse_with_item())

        self.expect_op(":")
        body = self.parse_suite()
        node = make_node("With", tok.lineno, tok.col, {"items": items, "body": body})
        if len(body) > 0:
            end_from_node(node, body[len(body) - 1])
        return node

    def parse_with_item(self) -> ASTNode:
        """Parse a single with item."""
        context_expr = self.parse_test()
        optional_vars: ASTNode | None = None
        if self.match("as"):
            self.advance()
            optional_vars = self.parse_exprlist_single()
            set_context(optional_vars, "Store")
        return {
            "_type": "withitem",
            "context_expr": context_expr,
            "optional_vars": optional_vars,
        }

    def parse_match_stmt(self) -> ASTNode:
        """Parse match statement."""
        tok = self.expect("match")
        subject = self.parse_test()
        self.expect_op(":")
        self.expect(TK_NEWLINE)
        self.expect(TK_INDENT)

        cases: list[ASTNode] = []
        while self.match("case"):
            cases.append(self.parse_case())
            self.skip_newlines()

        self.expect(TK_DEDENT)
        node = make_node(
            "Match", tok.lineno, tok.col, {"subject": subject, "cases": cases}
        )
        if len(cases) > 0:
            last_case = cases[len(cases) - 1]
            case_body = last_case.get("body")
            if isinstance(case_body, list) and len(case_body) > 0:
                end_from_node(node, case_body[len(case_body) - 1])
        return node

    def parse_case(self) -> ASTNode:
        """Parse a case clause."""
        self.expect("case")
        pattern = self.parse_pattern()
        guard: ASTNode | None = None
        if self.match("if"):
            self.advance()
            guard = self.parse_test()
        self.expect_op(":")
        body = self.parse_suite()
        return {"_type": "match_case", "pattern": pattern, "guard": guard, "body": body}

    def parse_pattern(self) -> ASTNode:
        """Parse a match pattern."""
        return self.parse_or_pattern()

    def parse_or_pattern(self) -> ASTNode:
        """Parse or pattern: pattern | pattern | ..."""
        patterns: list[ASTNode] = []
        patterns.append(self.parse_as_pattern())
        while self.match_op("|"):
            self.advance()
            patterns.append(self.parse_as_pattern())
        if len(patterns) == 1:
            return patterns[0]
        return {"_type": "MatchOr", "patterns": patterns}

    def parse_as_pattern(self) -> ASTNode:
        """Parse as pattern: pattern as name."""
        pattern = self.parse_closed_pattern()
        if self.match("as"):
            self.advance()
            name_tok = self.expect(TK_NAME)
            return {"_type": "MatchAs", "pattern": pattern, "name": name_tok.value}
        return pattern

    def parse_closed_pattern(self) -> ASTNode:
        """Parse a closed pattern."""
        tok = self.current()

        # Literal patterns
        if tok.type == TK_NUMBER:
            self.advance()
            return {"_type": "MatchValue", "value": make_constant_from_token(tok)}
        if tok.type == TK_STRING:
            self.advance()
            return {"_type": "MatchValue", "value": make_constant_from_token(tok)}
        if tok.value == "None":
            self.advance()
            return {"_type": "MatchSingleton", "value": None}
        if tok.value == "True":
            self.advance()
            return {"_type": "MatchSingleton", "value": True}
        if tok.value == "False":
            self.advance()
            return {"_type": "MatchSingleton", "value": False}

        # Negative numbers
        if self.match_op("-"):
            self.advance()
            num_tok = self.expect(TK_NUMBER)
            const = make_constant_from_token(num_tok)
            neg = end_from_node(
                make_node(
                    "UnaryOp",
                    tok.lineno,
                    tok.col,
                    {"op": {"_type": "USub"}, "operand": const},
                ),
                const,
            )
            return {"_type": "MatchValue", "value": neg}

        # Wildcard
        if tok.type == TK_NAME and tok.value == "_":
            self.advance()
            return {"_type": "MatchAs", "pattern": None, "name": None}

        # Capture pattern or class pattern
        if tok.type == TK_NAME:
            name = self.parse_dotted_name_for_pattern()
            if self.match_op("("):
                # Class pattern
                return self.parse_class_pattern(name, tok)
            # Check if it's an attribute pattern (MatchValue with dotted name)
            if "." in name:
                parts = name.split(".")
                result: ASTNode = end_from_token(
                    make_node(
                        "Name",
                        tok.lineno,
                        tok.col,
                        {"id": parts[0], "ctx": {"_type": "Load"}},
                    ),
                    self.prev_token(),
                )
                i = 1
                while i < len(parts):
                    result = end_from_token(
                        make_node(
                            "Attribute",
                            tok.lineno,
                            tok.col,
                            {
                                "value": result,
                                "attr": parts[i],
                                "ctx": {"_type": "Load"},
                            },
                        ),
                        self.prev_token(),
                    )
                    i += 1
                return {"_type": "MatchValue", "value": result}
            # Capture pattern
            return {"_type": "MatchAs", "pattern": None, "name": name}

        # Sequence pattern
        if self.match_op("["):
            return self.parse_sequence_pattern()
        if self.match_op("("):
            return self.parse_group_or_sequence_pattern()

        # Mapping pattern
        if self.match_op("{"):
            return self.parse_mapping_pattern()

        raise self.error("unexpected token '" + tok.value + "'")

    def parse_dotted_name_for_pattern(self) -> str:
        """Parse dotted name for pattern matching."""
        parts: list[str] = []
        parts.append(self.expect(TK_NAME).value)
        while self.match_op("."):
            self.advance()
            parts.append(self.expect(TK_NAME).value)
        return ".".join(parts)

    def parse_class_pattern(self, name: str, tok: Token) -> ASTNode:
        """Parse class pattern: Cls(patterns)."""
        self.expect_op("(")
        patterns: list[ASTNode] = []
        kwd_attrs: list[str] = []
        kwd_patterns: list[ASTNode] = []

        while not self.match_op(")"):
            if self.match_op(","):
                self.advance()
                continue

            # Check for keyword pattern: name=pattern
            if self.match(TK_NAME) and self.peek(1).value == "=":
                kwd_attrs.append(self.expect(TK_NAME).value)
                self.expect_op("=")
                kwd_patterns.append(self.parse_pattern())
            else:
                patterns.append(self.parse_pattern())

        self.expect_op(")")

        # Build class reference
        parts = name.split(".")
        cls: ASTNode = end_from_token(
            make_node(
                "Name", tok.lineno, tok.col, {"id": parts[0], "ctx": {"_type": "Load"}}
            ),
            tok,
        )
        i = 1
        while i < len(parts):
            cls = end_from_token(
                make_node(
                    "Attribute",
                    tok.lineno,
                    tok.col,
                    {"value": cls, "attr": parts[i], "ctx": {"_type": "Load"}},
                ),
                tok,
            )
            i += 1

        return {
            "_type": "MatchClass",
            "cls": cls,
            "patterns": patterns,
            "kwd_attrs": kwd_attrs,
            "kwd_patterns": kwd_patterns,
        }

    def parse_sequence_pattern(self) -> ASTNode:
        """Parse sequence pattern: [p1, p2, ...]."""
        self.expect_op("[")
        patterns: list[ASTNode] = []
        while not self.match_op("]"):
            if self.match_op(","):
                self.advance()
                continue
            if self.match_op("*"):
                self.advance()
                name: str | None = None
                if self.match(TK_NAME):
                    name = self.expect(TK_NAME).value
                patterns.append({"_type": "MatchStar", "name": name})
            else:
                patterns.append(self.parse_pattern())
        self.expect_op("]")
        return {"_type": "MatchSequence", "patterns": patterns}

    def parse_group_or_sequence_pattern(self) -> ASTNode:
        """Parse grouped or tuple pattern: (p) or (p1, p2, ...)."""
        self.expect_op("(")
        if self.match_op(")"):
            self.advance()
            return {"_type": "MatchSequence", "patterns": []}

        first = self.parse_pattern()
        if self.match_op(")"):
            self.advance()
            return first

        # It's a sequence
        patterns: list[ASTNode] = [first]
        while self.match_op(","):
            self.advance()
            if self.match_op(")"):
                break
            patterns.append(self.parse_pattern())
        self.expect_op(")")
        return {"_type": "MatchSequence", "patterns": patterns}

    def parse_mapping_pattern(self) -> ASTNode:
        """Parse mapping pattern: {k: v, ...}."""
        self.expect_op("{")
        keys: list[ASTNode] = []
        patterns: list[ASTNode] = []
        rest: str | None = None

        while not self.match_op("}"):
            if self.match_op(","):
                self.advance()
                continue
            if self.match_op("**"):
                self.advance()
                rest = self.expect(TK_NAME).value
            else:
                # Key can be a literal or dotted name
                key = self.parse_pattern_key()
                keys.append(key)
                self.expect_op(":")
                patterns.append(self.parse_pattern())

        self.expect_op("}")
        return {
            "_type": "MatchMapping",
            "keys": keys,
            "patterns": patterns,
            "rest": rest,
        }

    def parse_pattern_key(self) -> ASTNode:
        """Parse a key in mapping pattern."""
        tok = self.current()
        if tok.type == TK_NUMBER:
            self.advance()
            return make_constant_from_token(tok)
        if tok.type == TK_STRING:
            self.advance()
            return make_constant_from_token(tok)
        if tok.value in ("None", "True", "False"):
            self.advance()
            if tok.value == "None":
                return end_from_token(
                    make_node("Constant", tok.lineno, tok.col, {"value": None}), tok
                )
            if tok.value == "True":
                return end_from_token(
                    make_node("Constant", tok.lineno, tok.col, {"value": True}), tok
                )
            return end_from_token(
                make_node("Constant", tok.lineno, tok.col, {"value": False}), tok
            )
        if self.match_op("-"):
            self.advance()
            num_tok = self.expect(TK_NUMBER)
            const = make_constant_from_token(num_tok)
            return end_from_node(
                make_node(
                    "UnaryOp",
                    tok.lineno,
                    tok.col,
                    {"op": {"_type": "USub"}, "operand": const},
                ),
                const,
            )
        # Dotted name for attribute
        name = self.parse_dotted_name_for_pattern()
        parts = name.split(".")
        result: ASTNode = end_from_token(
            make_node(
                "Name", tok.lineno, tok.col, {"id": parts[0], "ctx": {"_type": "Load"}}
            ),
            self.prev_token(),
        )
        i = 1
        while i < len(parts):
            result = end_from_token(
                make_node(
                    "Attribute",
                    tok.lineno,
                    tok.col,
                    {"value": result, "attr": parts[i], "ctx": {"_type": "Load"}},
                ),
                self.prev_token(),
            )
            i += 1
        return result

    def parse_suite(self) -> list[ASTNode]:
        """Parse a suite (block of statements)."""
        # Simple suite on same line
        if not self.match(TK_NEWLINE):
            stmt = self.parse_simple_stmt()
            if stmt is None:
                return []
            if isinstance(stmt, list):
                return stmt
            return [stmt]

        # Indented block
        self.expect(TK_NEWLINE)
        self.expect(TK_INDENT)
        stmts: list[ASTNode] = []
        while not self.match(TK_DEDENT) and not self.match(TK_ENDMARKER):
            self.skip_newlines()
            if self.match(TK_DEDENT) or self.match(TK_ENDMARKER):
                break
            stmt = self.parse_stmt()
            if stmt is not None:
                if isinstance(stmt, list):
                    i = 0
                    while i < len(stmt):
                        stmts.append(stmt[i])
                        i += 1
                else:
                    stmts.append(stmt)
        if self.match(TK_DEDENT):
            self.advance()
        return stmts

    # --- Expression parsing ---

    def parse_namedexpr_test(self) -> ASTNode:
        """Parse test with optional walrus operator: test [':=' test]."""
        tok = self.current()
        expr = self.parse_test()
        if self.match_op(":="):
            self.advance()
            value = self.parse_test()
            validate_target(expr, "Store", False, True, False)
            self.check_walrus_scope(expr)
            if "ctx" in expr:
                expr["ctx"] = {"_type": "Store"}
            return end_from_node(
                make_node(
                    "NamedExpr",
                    tok.lineno,
                    tok.col,
                    {"target": expr, "value": value},
                ),
                value,
            )
        return expr

    def parse_test(self) -> ASTNode:
        """Parse test expression (may be ternary or lambda)."""
        tok = self.current()

        # Lambda
        if tok.value == "lambda":
            return self.parse_lambda()

        # Or test
        expr = self.parse_or_test()

        # Ternary: expr if test else expr
        if self.match("if"):
            self.advance()
            condition = self.parse_or_test()
            self.expect("else")
            orelse = self.parse_test()
            return end_from_node(
                make_node(
                    "IfExp",
                    tok.lineno,
                    tok.col,
                    {"test": condition, "body": expr, "orelse": orelse},
                ),
                orelse,
            )

        return expr

    def parse_lambda(self) -> ASTNode:
        """Parse lambda expression."""
        tok = self.expect("lambda")
        params = make_arguments()
        if not self.match_op(":"):
            params = self.parse_varargslist()
        self.expect_op(":")
        body = self.parse_test()
        return end_from_node(
            make_node("Lambda", tok.lineno, tok.col, {"args": params, "body": body}),
            body,
        )

    def parse_varargslist(self) -> ASTNode:
        """Parse lambda argument list (no type annotations)."""
        args: list[ASTNode] = []
        posonlyargs: list[ASTNode] = []
        kwonlyargs: list[ASTNode] = []
        defaults: list[ASTNode] = []
        kw_defaults: list[ASTNode | None] = []
        vararg: ASTNode | None = None
        kwarg: ASTNode | None = None
        in_kwonly = False
        has_default = False

        while not self.match_op(":"):
            if self.match_op(","):
                self.advance()
                continue

            if self.match_op("/"):
                self.advance()
                posonlyargs = args[:]
                args = []
                has_default = False
                continue

            if self.match_op("*"):
                self.advance()
                in_kwonly = True
                if self.match(TK_NAME):
                    tok = self.expect(TK_NAME)
                    vararg = {
                        "_type": "arg",
                        "arg": tok.value,
                        "annotation": None,
                        "lineno": tok.lineno,
                        "col_offset": tok.col,
                        "end_lineno": tok.lineno,
                        "end_col_offset": tok.col + len(tok.value),
                    }
                continue

            if self.match_op("**"):
                self.advance()
                tok = self.expect(TK_NAME)
                kwarg = {
                    "_type": "arg",
                    "arg": tok.value,
                    "annotation": None,
                    "lineno": tok.lineno,
                    "col_offset": tok.col,
                    "end_lineno": tok.lineno,
                    "end_col_offset": tok.col + len(tok.value),
                }
                continue

            tok = self.expect(TK_NAME)
            arg: ASTNode = {
                "_type": "arg",
                "arg": tok.value,
                "annotation": None,
                "lineno": tok.lineno,
                "col_offset": tok.col,
                "end_lineno": tok.lineno,
                "end_col_offset": tok.col + len(tok.value),
            }
            default: ASTNode | None = None
            if self.match_op("="):
                self.advance()
                default = self.parse_test()

            if in_kwonly:
                kwonlyargs.append(arg)
                kw_defaults.append(default)
            else:
                if default is not None:
                    has_default = True
                elif has_default:
                    raise self.error("non-default argument follows default argument")
                args.append(arg)
                if default is not None:
                    defaults.append(default)

        return {
            "_type": "arguments",
            "posonlyargs": posonlyargs,
            "args": args,
            "vararg": vararg,
            "kwonlyargs": kwonlyargs,
            "kw_defaults": kw_defaults,
            "kwarg": kwarg,
            "defaults": defaults,
        }

    def parse_or_test(self) -> ASTNode:
        """Parse or_test: and_test ('or' and_test)*."""
        tok = self.current()
        values: list[ASTNode] = []
        values.append(self.parse_and_test())
        while self.match("or"):
            self.advance()
            values.append(self.parse_and_test())
        if len(values) == 1:
            return values[0]
        return end_from_node(
            make_node(
                "BoolOp", tok.lineno, tok.col, {"op": {"_type": "Or"}, "values": values}
            ),
            values[len(values) - 1],
        )

    def parse_and_test(self) -> ASTNode:
        """Parse and_test: not_test ('and' not_test)*."""
        tok = self.current()
        values: list[ASTNode] = []
        values.append(self.parse_not_test())
        while self.match("and"):
            self.advance()
            values.append(self.parse_not_test())
        if len(values) == 1:
            return values[0]
        return end_from_node(
            make_node(
                "BoolOp",
                tok.lineno,
                tok.col,
                {"op": {"_type": "And"}, "values": values},
            ),
            values[len(values) - 1],
        )

    def parse_not_test(self) -> ASTNode:
        """Parse not_test: 'not' not_test | comparison."""
        tok = self.current()
        if self.match("not"):
            self.advance()
            operand = self.parse_not_test()
            return end_from_node(
                make_node(
                    "UnaryOp",
                    tok.lineno,
                    tok.col,
                    {"op": {"_type": "Not"}, "operand": operand},
                ),
                operand,
            )
        return self.parse_comparison()

    def parse_comparison(self) -> ASTNode:
        """Parse comparison: expr (comp_op expr)*."""
        tok = self.current()
        left = self.parse_expr()
        ops: list[ASTNode] = []
        comparators: list[ASTNode] = []

        while True:
            op = self.parse_comp_op()
            if op is None:
                break
            ops.append(op)
            comparators.append(self.parse_expr())

        if len(ops) == 0:
            return left
        return end_from_node(
            make_node(
                "Compare",
                tok.lineno,
                tok.col,
                {"left": left, "ops": ops, "comparators": comparators},
            ),
            comparators[len(comparators) - 1],
        )

    def parse_comp_op(self) -> ASTNode | None:
        """Parse comparison operator."""
        tok = self.current()
        if self.match_op("<"):
            self.advance()
            return {"_type": "Lt"}
        if self.match_op(">"):
            self.advance()
            return {"_type": "Gt"}
        if self.match_op("=="):
            self.advance()
            return {"_type": "Eq"}
        if self.match_op(">="):
            self.advance()
            return {"_type": "GtE"}
        if self.match_op("<="):
            self.advance()
            return {"_type": "LtE"}
        if self.match_op("!="):
            self.advance()
            return {"_type": "NotEq"}
        if self.match("in"):
            self.advance()
            return {"_type": "In"}
        if self.match("not"):
            self.advance()
            self.expect("in")
            return {"_type": "NotIn"}
        if self.match("is"):
            self.advance()
            if self.match("not"):
                self.advance()
                return {"_type": "IsNot"}
            return {"_type": "Is"}
        return None

    def parse_expr(self) -> ASTNode:
        """Parse expr: xor_expr ('|' xor_expr)*."""
        tok = self.current()
        left = self.parse_xor_expr()
        while self.match_op("|"):
            self.advance()
            right = self.parse_xor_expr()
            left = end_from_node(
                make_node(
                    "BinOp",
                    tok.lineno,
                    tok.col,
                    {"left": left, "op": {"_type": "BitOr"}, "right": right},
                ),
                right,
            )
        return left

    def parse_xor_expr(self) -> ASTNode:
        """Parse xor_expr: and_expr ('^' and_expr)*."""
        tok = self.current()
        left = self.parse_and_expr()
        while self.match_op("^"):
            self.advance()
            right = self.parse_and_expr()
            left = end_from_node(
                make_node(
                    "BinOp",
                    tok.lineno,
                    tok.col,
                    {"left": left, "op": {"_type": "BitXor"}, "right": right},
                ),
                right,
            )
        return left

    def parse_and_expr(self) -> ASTNode:
        """Parse and_expr: shift_expr ('&' shift_expr)*."""
        tok = self.current()
        left = self.parse_shift_expr()
        while self.match_op("&"):
            self.advance()
            right = self.parse_shift_expr()
            left = end_from_node(
                make_node(
                    "BinOp",
                    tok.lineno,
                    tok.col,
                    {"left": left, "op": {"_type": "BitAnd"}, "right": right},
                ),
                right,
            )
        return left

    def parse_shift_expr(self) -> ASTNode:
        """Parse shift_expr: arith_expr (('<<'|'>>') arith_expr)*."""
        tok = self.current()
        left = self.parse_arith_expr()
        while self.match_op("<<") or self.match_op(">>"):
            op_tok = self.advance()
            op_type = "LShift" if op_tok.value == "<<" else "RShift"
            right = self.parse_arith_expr()
            left = end_from_node(
                make_node(
                    "BinOp",
                    tok.lineno,
                    tok.col,
                    {"left": left, "op": {"_type": op_type}, "right": right},
                ),
                right,
            )
        return left

    def parse_arith_expr(self) -> ASTNode:
        """Parse arith_expr: term (('+'|'-') term)*."""
        tok = self.current()
        left = self.parse_term()
        while self.match_op("+") or self.match_op("-"):
            op_tok = self.advance()
            op_type = "Add" if op_tok.value == "+" else "Sub"
            right = self.parse_term()
            left = end_from_node(
                make_node(
                    "BinOp",
                    tok.lineno,
                    tok.col,
                    {"left": left, "op": {"_type": op_type}, "right": right},
                ),
                right,
            )
        return left

    def parse_term(self) -> ASTNode:
        """Parse term: factor (('*'|'/'|'//'|'%'|'@') factor)*."""
        tok = self.current()
        left = self.parse_factor()
        while True:
            op_type: str | None = None
            if self.match_op("*"):
                op_type = "Mult"
            elif self.match_op("/"):
                op_type = "Div"
            elif self.match_op("//"):
                op_type = "FloorDiv"
            elif self.match_op("%"):
                op_type = "Mod"
            elif self.match_op("@"):
                op_type = "MatMult"
            else:
                break
            self.advance()
            right = self.parse_factor()
            left = end_from_node(
                make_node(
                    "BinOp",
                    tok.lineno,
                    tok.col,
                    {"left": left, "op": {"_type": op_type}, "right": right},
                ),
                right,
            )
        return left

    def parse_factor(self) -> ASTNode:
        """Parse factor: ('+'|'-'|'~') factor | power."""
        tok = self.current()
        if self.match_op("+"):
            self.advance()
            operand = self.parse_factor()
            return end_from_node(
                make_node(
                    "UnaryOp",
                    tok.lineno,
                    tok.col,
                    {"op": {"_type": "UAdd"}, "operand": operand},
                ),
                operand,
            )
        if self.match_op("-"):
            self.advance()
            operand = self.parse_factor()
            return end_from_node(
                make_node(
                    "UnaryOp",
                    tok.lineno,
                    tok.col,
                    {"op": {"_type": "USub"}, "operand": operand},
                ),
                operand,
            )
        if self.match_op("~"):
            self.advance()
            operand = self.parse_factor()
            return end_from_node(
                make_node(
                    "UnaryOp",
                    tok.lineno,
                    tok.col,
                    {"op": {"_type": "Invert"}, "operand": operand},
                ),
                operand,
            )
        return self.parse_power()

    def parse_power(self) -> ASTNode:
        """Parse power: await_expr ['**' factor]."""
        tok = self.current()
        base = self.parse_await_expr()
        if self.match_op("**"):
            self.advance()
            exp = self.parse_factor()
            return end_from_node(
                make_node(
                    "BinOp",
                    tok.lineno,
                    tok.col,
                    {"left": base, "op": {"_type": "Pow"}, "right": exp},
                ),
                exp,
            )
        return base

    def parse_await_expr(self) -> ASTNode:
        """Parse await_expr: ['await'] atom_expr."""
        tok = self.current()
        if self.match("await"):
            if self.async_depth == 0:
                raise self.error("'await' outside async function")
            self.advance()
            value = self.parse_atom_expr()
            return end_from_node(
                make_node("Await", tok.lineno, tok.col, {"value": value}), value
            )
        return self.parse_atom_expr()

    def parse_atom_expr(self) -> ASTNode:
        """Parse atom_expr: atom trailer*."""
        base = self.parse_atom()
        while True:
            if self.match_op("("):
                base = self.parse_call(base)
            elif self.match_op("["):
                base = self.parse_subscript(base)
            elif self.match_op("."):
                base = self.parse_attribute(base)
            else:
                break
        return base

    def parse_call(self, func: ASTNode) -> ASTNode:
        """Parse function call trailer."""
        tok = self.expect_op("(")
        args: list[ASTNode] = []
        keywords: list[ASTNode] = []

        if not self.match_op(")"):
            args, keywords = self.parse_arglist()

        close = self.expect_op(")")
        return end_from_token(
            make_node(
                "Call",
                tok.lineno,
                tok.col,
                {"func": func, "args": args, "keywords": keywords},
            ),
            close,
        )

    def parse_arglist(self) -> tuple[list[ASTNode], list[ASTNode]]:
        """Parse argument list."""
        args: list[ASTNode] = []
        keywords: list[ASTNode] = []
        has_keyword = False
        has_kwargs = False
        seen_keywords: set[str] = set()

        while not self.match_op(")"):
            if self.match_op(","):
                self.advance()
                continue

            # **kwargs
            if self.match_op("**"):
                self.advance()
                value = self.parse_test()
                keywords.append({"_type": "keyword", "arg": None, "value": value})
                has_kwargs = True
                continue

            # *args
            if self.match_op("*"):
                if has_kwargs:
                    raise self.error("iterable argument unpacking follows keyword argument unpacking")
                star_tok = self.advance()
                value = self.parse_test()
                args.append(
                    end_from_node(
                        make_node(
                            "Starred",
                            star_tok.lineno,
                            star_tok.col,
                            {"value": value, "ctx": {"_type": "Load"}},
                        ),
                        value,
                    )
                )
                continue

            # Check for keyword argument
            if self.match(TK_NAME) and self.peek(1).value == "=":
                name = self.expect(TK_NAME).value
                self.expect_op("=")
                value = self.parse_test()
                if name in seen_keywords:
                    raise self.error("keyword argument repeated: " + name)
                seen_keywords.add(name)
                keywords.append({"_type": "keyword", "arg": name, "value": value})
                has_keyword = True
                continue

            # Positional after keyword/kwargs is an error
            if has_kwargs:
                raise self.error("positional argument follows keyword argument unpacking")
            if has_keyword:
                raise self.error("positional argument follows keyword argument")

            # Positional argument (may include comprehension)
            arg = self.parse_test()

            # Check for comprehension making this a generator expression
            if self.match("for"):
                generators = self.parse_comp_for()
                _check_comp_walrus([arg], generators, self.class_depth > 0)
                arg = end_from_token(
                    make_node(
                        "GeneratorExp",
                        arg.get("lineno", 1),
                        arg.get("col_offset", 0),
                        {"elt": arg, "generators": generators},
                    ),
                    self.prev_token(),
                )
                if len(args) > 0 or not self.match_op(")"):
                    raise self.error(
                        "generator expression must be parenthesized"
                    )

            args.append(arg)

        return args, keywords

    def parse_comp_for(self) -> list[ASTNode]:
        """Parse comprehension for clause(s)."""
        generators: list[ASTNode] = []
        saved_comp_depth = self.comp_depth
        saved_in_class_comp = self.in_class_comp
        self.comp_depth += 1
        target_names: set[str] = set()
        self.comp_target_names.append(target_names)
        if self.class_depth > 0 and saved_comp_depth == 0:
            self.in_class_comp = True
        while self.match("for") or self.match("async"):
            is_async = 0
            if self.match("async"):
                self.advance()
                is_async = 1
            self.expect("for")
            target = self.parse_target_list()
            if isinstance(target, dict) and target.get("_type") == "Starred":
                raise self.error("starred assignment target must be in a list or tuple")
            set_context(target, "Store")
            _collect_names(target, target_names)
            self.expect("in")
            self.comp_iter_depth += 1
            iter_expr = self.parse_or_test()
            self.comp_iter_depth -= 1
            ifs: list[ASTNode] = []
            while self.match("if"):
                self.advance()
                ifs.append(self.parse_or_test())
            generators.append(
                {
                    "_type": "comprehension",
                    "target": target,
                    "iter": iter_expr,
                    "ifs": ifs,
                    "is_async": is_async,
                }
            )
        self.comp_target_names.pop()
        self.comp_depth = saved_comp_depth
        self.in_class_comp = saved_in_class_comp
        return generators

    def parse_target_list(self) -> ASTNode:
        """Parse target list for for/comprehension (stops at 'in')."""
        tok = self.current()
        items: list[ASTNode] = []
        items.append(self.parse_target())
        while self.match_op(","):
            self.advance()
            if self.match("in"):
                break
            items.append(self.parse_target())
        if len(items) == 1:
            return items[0]
        return end_from_node(
            make_node(
                "Tuple", tok.lineno, tok.col, {"elts": items, "ctx": {"_type": "Load"}}
            ),
            items[len(items) - 1],
        )

    def parse_target(self) -> ASTNode:
        """Parse a single target (name, attribute, subscript, or tuple/list)."""
        tok = self.current()
        if self.match_op("("):
            self.advance()
            if self.match_op(")"):
                close = self.advance()
                return end_from_token(
                    make_node(
                        "Tuple",
                        tok.lineno,
                        tok.col,
                        {"elts": [], "ctx": {"_type": "Load"}},
                    ),
                    close,
                )
            inner = self.parse_target_list()
            self.expect_op(")")
            return inner
        if self.match_op("["):
            self.advance()
            if self.match_op("]"):
                close = self.advance()
                return end_from_token(
                    make_node(
                        "List",
                        tok.lineno,
                        tok.col,
                        {"elts": [], "ctx": {"_type": "Load"}},
                    ),
                    close,
                )
            items: list[ASTNode] = []
            items.append(self.parse_target())
            while self.match_op(","):
                self.advance()
                if self.match_op("]"):
                    break
                items.append(self.parse_target())
            close = self.expect_op("]")
            return end_from_token(
                make_node(
                    "List",
                    tok.lineno,
                    tok.col,
                    {"elts": items, "ctx": {"_type": "Load"}},
                ),
                close,
            )
        if self.match_op("*"):
            star_tok = self.advance()
            value = self.parse_target()
            return end_from_node(
                make_node(
                    "Starred",
                    star_tok.lineno,
                    star_tok.col,
                    {"value": value, "ctx": {"_type": "Load"}},
                ),
                value,
            )
        # Name with optional attribute/subscript
        base = self.parse_atom()
        while True:
            if self.match_op("."):
                base = self.parse_attribute(base)
            elif self.match_op("["):
                base = self.parse_subscript(base)
            else:
                break
        return base

    def parse_subscript(self, value: ASTNode) -> ASTNode:
        """Parse subscript trailer."""
        tok = self.expect_op("[")
        slice_node = self.parse_subscript_inner()
        close = self.expect_op("]")
        return end_from_token(
            make_node(
                "Subscript",
                tok.lineno,
                tok.col,
                {"value": value, "slice": slice_node, "ctx": {"_type": "Load"}},
            ),
            close,
        )

    def parse_subscript_inner(self) -> ASTNode:
        """Parse subscript content (may be slice or tuple of slices)."""
        items: list[ASTNode] = []
        items.append(self.parse_subscript_item())
        while self.match_op(","):
            self.advance()
            if self.match_op("]"):
                break
            items.append(self.parse_subscript_item())
        if len(items) == 1:
            return items[0]
        return end_from_node(
            make_node(
                "Tuple",
                self.current().lineno,
                self.current().col,
                {"elts": items, "ctx": {"_type": "Load"}},
            ),
            items[len(items) - 1],
        )

    def parse_subscript_item(self) -> ASTNode:
        """Parse a single subscript item (slice or expression)."""
        tok = self.current()

        # Check for slice
        lower: ASTNode | None = None
        upper: ASTNode | None = None
        step: ASTNode | None = None

        if not self.match_op(":"):
            lower = self.parse_test()
            if not self.match_op(":"):
                return lower

        # First colon
        self.expect_op(":")

        if not self.match_op(":") and not self.match_op(",") and not self.match_op("]"):
            upper = self.parse_test()

        # Optional second colon for step
        if self.match_op(":"):
            self.advance()
            if not self.match_op(",") and not self.match_op("]"):
                step = self.parse_test()

        node = make_node(
            "Slice", tok.lineno, tok.col, {"lower": lower, "upper": upper, "step": step}
        )
        if step is not None:
            end_from_node(node, step)
        elif upper is not None:
            end_from_node(node, upper)
        elif lower is not None:
            end_from_token(node, self.prev_token())
        else:
            end_from_token(node, self.prev_token())
        return node

    def parse_attribute(self, value: ASTNode) -> ASTNode:
        """Parse attribute access trailer."""
        tok = self.expect_op(".")
        name_tok = self.expect(TK_NAME)
        return end_from_token(
            make_node(
                "Attribute",
                tok.lineno,
                tok.col,
                {"value": value, "attr": name_tok.value, "ctx": {"_type": "Load"}},
            ),
            name_tok,
        )

    def parse_atom(self) -> ASTNode:
        """Parse atom: literal, name, or parenthesized expression."""
        tok = self.current()

        # Parenthesized expression, tuple, or generator
        if self.match_op("("):
            self.advance()
            if self.match_op(")"):
                close = self.advance()
                return end_from_token(
                    make_node(
                        "Tuple",
                        tok.lineno,
                        tok.col,
                        {"elts": [], "ctx": {"_type": "Load"}},
                    ),
                    close,
                )

            # Check for yield
            if self.match("yield"):
                expr = self.parse_yield_expr()
                self.expect_op(")")
                return expr

            first = self.parse_testlist_star_expr_item()

            # Named expression (walrus operator)
            if self.match_op(":="):
                self.advance()
                value = self.parse_test()
                validate_target(first, "Store", False, True, False)
                self.check_walrus_scope(first)
                if "ctx" in first:
                    first["ctx"] = {"_type": "Store"}
                first = end_from_node(
                    make_node(
                        "NamedExpr",
                        tok.lineno,
                        tok.col,
                        {"target": first, "value": value},
                    ),
                    value,
                )
                if self.match_op(")"):
                    self.advance()
                    return first

            # Generator expression
            if self.match("for"):
                generators = self.parse_comp_for()
                _check_comp_walrus([first], generators, self.class_depth > 0)
                close = self.expect_op(")")
                return end_from_token(
                    make_node(
                        "GeneratorExp",
                        tok.lineno,
                        tok.col,
                        {"elt": first, "generators": generators},
                    ),
                    close,
                )

            # Tuple or single expression
            if self.match_op(","):
                elts: list[ASTNode] = [first]
                while self.match_op(","):
                    self.advance()
                    if self.match_op(")"):
                        break
                    elts.append(self.parse_testlist_star_expr_item())
                close = self.expect_op(")")
                return end_from_token(
                    make_node(
                        "Tuple",
                        tok.lineno,
                        tok.col,
                        {"elts": elts, "ctx": {"_type": "Load"}},
                    ),
                    close,
                )

            if isinstance(first, dict) and first.get("_type") == "Starred":
                raise self.error("starred expression is not allowed here")
            self.expect_op(")")
            return first

        # List
        if self.match_op("["):
            self.advance()
            if self.match_op("]"):
                close = self.advance()
                return end_from_token(
                    make_node(
                        "List",
                        tok.lineno,
                        tok.col,
                        {"elts": [], "ctx": {"_type": "Load"}},
                    ),
                    close,
                )

            first = self.parse_testlist_star_expr_item()

            # List comprehension
            if self.match("for"):
                if isinstance(first, dict) and first.get("_type") == "Starred":
                    raise self.error("iterable unpacking cannot be used in comprehension")
                generators = self.parse_comp_for()
                _check_comp_walrus([first], generators, self.class_depth > 0)
                close = self.expect_op("]")
                return end_from_token(
                    make_node(
                        "ListComp",
                        tok.lineno,
                        tok.col,
                        {"elt": first, "generators": generators},
                    ),
                    close,
                )

            # Regular list
            elts: list[ASTNode] = [first]
            while self.match_op(","):
                self.advance()
                if self.match_op("]"):
                    break
                elts.append(self.parse_testlist_star_expr_item())
            close = self.expect_op("]")
            return end_from_token(
                make_node(
                    "List",
                    tok.lineno,
                    tok.col,
                    {"elts": elts, "ctx": {"_type": "Load"}},
                ),
                close,
            )

        # Dict or set
        if self.match_op("{"):
            return self.parse_dict_or_set()

        # Name
        if self.match(TK_NAME):
            name_tok = self.advance()
            return end_from_token(
                make_node(
                    "Name",
                    name_tok.lineno,
                    name_tok.col,
                    {"id": name_tok.value, "ctx": {"_type": "Load"}},
                ),
                name_tok,
            )

        # Number
        if self.match(TK_NUMBER):
            num_tok = self.advance()
            return make_constant_from_token(num_tok)

        # String(s)
        if self.match(TK_STRING):
            return self.parse_strings()

        # None, True, False
        if self.match("None"):
            self.advance()
            return end_from_token(
                make_node("Constant", tok.lineno, tok.col, {"value": None}), tok
            )
        if self.match("True"):
            self.advance()
            return end_from_token(
                make_node("Constant", tok.lineno, tok.col, {"value": True}), tok
            )
        if self.match("False"):
            self.advance()
            return end_from_token(
                make_node("Constant", tok.lineno, tok.col, {"value": False}), tok
            )

        # Ellipsis
        if self.match_op("..."):
            self.advance()
            return end_from_token(
                make_node("Constant", tok.lineno, tok.col, {"value": ...}), tok
            )

        raise self.error("unexpected token '" + tok.value + "'")

    def parse_dict_or_set(self) -> ASTNode:
        """Parse dict or set literal."""
        tok = self.expect_op("{")

        if self.match_op("}"):
            close = self.advance()
            return end_from_token(
                make_node("Dict", tok.lineno, tok.col, {"keys": [], "values": []}),
                close,
            )

        # Check first item to determine if dict or set
        first = self.parse_dict_or_set_item()

        # Dict unpacking or dict
        if isinstance(first, tuple):
            # It's a dict
            keys: list[ASTNode | None] = []
            values: list[ASTNode] = []
            keys.append(first[0])
            values.append(first[1])

            # Check for dict comprehension
            if first[0] is not None and self.match("for"):
                generators = self.parse_comp_for()
                _check_comp_walrus([first[0], first[1]], generators, self.class_depth > 0)
                close = self.expect_op("}")
                return end_from_token(
                    make_node(
                        "DictComp",
                        tok.lineno,
                        tok.col,
                        {
                            "key": first[0],
                            "value": first[1],
                            "generators": generators,
                        },
                    ),
                    close,
                )

            while self.match_op(","):
                self.advance()
                if self.match_op("}"):
                    break
                item = self.parse_dict_or_set_item()
                if isinstance(item, tuple):
                    keys.append(item[0])
                    values.append(item[1])
                else:
                    # Mixing dict unpacking
                    keys.append(None)
                    values.append(item)
            close = self.expect_op("}")
            return end_from_token(
                make_node(
                    "Dict", tok.lineno, tok.col, {"keys": keys, "values": values}
                ),
                close,
            )

        # Set
        elts: list[ASTNode] = [first]

        # Check for set comprehension
        if self.match("for"):
            generators = self.parse_comp_for()
            _check_comp_walrus([first], generators, self.class_depth > 0)
            close = self.expect_op("}")
            return end_from_token(
                make_node(
                    "SetComp",
                    tok.lineno,
                    tok.col,
                    {"elt": first, "generators": generators},
                ),
                close,
            )

        while self.match_op(","):
            self.advance()
            if self.match_op("}"):
                break
            item = self.parse_dict_or_set_item()
            if isinstance(item, tuple):
                raise self.error("cannot mix dict and set syntax")
            elts.append(item)
        close = self.expect_op("}")
        return end_from_token(
            make_node("Set", tok.lineno, tok.col, {"elts": elts}), close
        )

    def parse_dict_or_set_item(self) -> ASTNode | tuple[ASTNode | None, ASTNode]:
        """Parse a dict or set item. Returns tuple for dict, ASTNode for set."""
        # Dict unpacking
        if self.match_op("**"):
            self.advance()
            value = self.parse_test()
            return (None, value)

        first = self.parse_test()

        # Dict key-value
        if self.match_op(":"):
            self.advance()
            value = self.parse_test()
            return (first, value)

        # Set element
        return first

    def parse_strings(self) -> ASTNode:
        """Parse one or more string literals (concatenation)."""
        tok = self.current()
        strings: list[Token] = []
        while self.match(TK_STRING):
            strings.append(self.advance())

        # Check for f-strings and string/bytes mixing
        has_fstring = False
        has_bytes = False
        has_str = False
        i = 0
        while i < len(strings):
            val = strings[i].value
            quote_pos = 0
            while quote_pos < len(val) and val[quote_pos] not in "\"'":
                quote_pos += 1
            prefix = val[:quote_pos].lower()
            if "f" in prefix:
                has_fstring = True
                has_str = True
            elif "b" in prefix:
                has_bytes = True
            else:
                has_str = True
            i += 1
        if has_bytes and (has_str or has_fstring):
            raise self.error("cannot mix bytes and nonbytes literals")

        if has_fstring:
            # Parse f-string content to extract literal parts and {expr} parts
            values: list[ASTNode] = []
            j = 0
            while j < len(strings):
                s = strings[j]
                fstring_values = parse_fstring(s.value, s.lineno, s.col)
                k = 0
                while k < len(fstring_values):
                    values.append(fstring_values[k])
                    k += 1
                j += 1
            last_str = strings[len(strings) - 1]
            return end_from_token(
                make_node("JoinedStr", tok.lineno, tok.col, {"values": values}),
                last_str,
            )

        # Regular strings - concatenate
        combined = parse_string_value(strings[0].value, strings[0].lineno, strings[0].col)
        k = 1
        while k < len(strings):
            next_val = parse_string_value(strings[k].value, strings[k].lineno, strings[k].col)
            if isinstance(combined, str) and isinstance(next_val, str):
                combined = combined + next_val
            elif isinstance(combined, bytes) and isinstance(next_val, bytes):
                combined = combined + next_val
            k += 1

        last_str = strings[len(strings) - 1]
        return end_from_token(
            make_node("Constant", tok.lineno, tok.col, {"value": combined}), last_str
        )

    def parse_testlist_star_expr(self) -> ASTNode:
        """Parse testlist_star_expr: (test|star_expr) (',' (test|star_expr))* [',']."""
        tok = self.current()
        items: list[ASTNode] = []
        items.append(self.parse_testlist_star_expr_item())

        has_comma = False
        while self.match_op(","):
            has_comma = True
            self.advance()
            if self.is_end_of_testlist():
                break
            items.append(self.parse_testlist_star_expr_item())

        if len(items) == 1 and not has_comma:
            return items[0]
        return end_from_node(
            make_node(
                "Tuple", tok.lineno, tok.col, {"elts": items, "ctx": {"_type": "Load"}}
            ),
            items[len(items) - 1],
        )

    def parse_testlist_star_expr_item(self) -> ASTNode:
        """Parse a single item in testlist_star_expr."""
        if self.match_op("*"):
            tok = self.advance()
            value = self.parse_test()
            return end_from_node(
                make_node(
                    "Starred",
                    tok.lineno,
                    tok.col,
                    {"value": value, "ctx": {"_type": "Load"}},
                ),
                value,
            )
        return self.parse_test()

    def parse_exprlist(self) -> list[ASTNode]:
        """Parse exprlist: expr (',' expr)* [',']."""
        items: list[ASTNode] = []
        items.append(self.parse_test())
        while self.match_op(","):
            self.advance()
            if self.is_end_of_testlist():
                break
            items.append(self.parse_test())
        return items

    def parse_exprlist_single(self) -> ASTNode:
        """Parse exprlist returning single node or tuple."""
        tok = self.current()
        items = self.parse_exprlist()
        if len(items) == 1:
            return items[0]
        return end_from_node(
            make_node(
                "Tuple", tok.lineno, tok.col, {"elts": items, "ctx": {"_type": "Load"}}
            ),
            items[len(items) - 1],
        )

    def is_end_of_testlist(self) -> bool:
        """Check if we're at the end of a testlist."""
        tok = self.current()
        if tok.type == TK_NEWLINE:
            return True
        if tok.type == TK_ENDMARKER:
            return True
        if tok.type == TK_OP and tok.value in (")", "]", "}", ":", ";", "="):
            return True
        if tok.value in ("for", "if", "async", "in"):
            return True
        return False


# --- Helper functions ---


def make_node(
    type_name: str, lineno: int, col: int, fields: dict[str, object] | None = None
) -> ASTNode:
    """Create an AST dict node with position info."""
    result: ASTNode = {"_type": type_name}
    result["lineno"] = lineno
    result["col_offset"] = col
    result["end_lineno"] = lineno
    result["end_col_offset"] = col
    if fields is not None:
        keys = list(fields.keys())
        i = 0
        while i < len(keys):
            key = keys[i]
            result[key] = fields[key]
            i += 1
    return result


def end_from_token(node: ASTNode, tok: Token) -> ASTNode:
    """Set end position from a token."""
    node["end_lineno"] = tok.lineno
    node["end_col_offset"] = tok.col + len(tok.value)
    return node


def end_from_node(node: ASTNode, child: ASTNode) -> ASTNode:
    """Set end position from a child node."""
    node["end_lineno"] = child.get("end_lineno", node.get("lineno", 1))
    node["end_col_offset"] = child.get("end_col_offset", node.get("col_offset", 0))
    return node


def make_arguments() -> ASTNode:
    """Create empty arguments node."""
    return {
        "_type": "arguments",
        "posonlyargs": [],
        "args": [],
        "vararg": None,
        "kwonlyargs": [],
        "kw_defaults": [],
        "kwarg": None,
        "defaults": [],
    }


def make_constant_from_token(tok: Token) -> ASTNode:
    """Create Constant node from number or string token with proper end position."""
    if tok.type == TK_NUMBER:
        value = parse_number_value(tok.value)
    else:
        value = parse_string_value(tok.value, tok.lineno, tok.col)
    node = make_node("Constant", tok.lineno, tok.col, {"value": value})
    node["end_col_offset"] = tok.col + len(tok.value)
    return node


def parse_number_value(s: str) -> int | float:
    """Parse a number literal string to value."""
    s = s.replace("_", "")
    if s.endswith(("j", "J")):
        return float(s[:-1])
    if "." in s or (
        "e" in s.lower() and not s.startswith(("0x", "0X", "0b", "0B", "0o", "0O"))
    ):
        return float(s)
    return int(s, 0)


def parse_string_value(s: str, lineno: int = 1, col: int = 0) -> str | bytes:
    """Parse a string literal to its value."""
    # Handle prefixes
    prefix = ""
    i = 0
    while i < len(s) and s[i] in "rRbBfFuU":
        prefix = prefix + s[i].lower()
        i += 1

    # Get quote style
    quote = s[i]
    if s[i : i + 3] in ('"""', "'''"):
        quote = s[i : i + 3]
        content = s[i + 3 : -3]
    else:
        content = s[i + 1 : -1]

    is_bytes = "b" in prefix

    # Validate bytes literal content
    if is_bytes and "r" not in prefix:
        j = 0
        while j < len(content):
            if content[j] == "\\" and j + 1 < len(content):
                j += 2
                continue
            if ord(content[j]) > 127:
                raise ParseError("bytes can only contain ASCII literal characters", lineno, col)
            j += 1

    # Handle raw strings
    if "r" in prefix:
        if is_bytes:
            return content.encode("latin-1")
        return content

    # Process escape sequences
    result = process_escapes(content, is_bytes, lineno, col)
    return result


def process_escapes(s: str, is_bytes: bool, lineno: int = 1, col: int = 0) -> str | bytes:
    """Process escape sequences in string."""
    result: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            next_c = s[i + 1]
            if next_c == "n":
                result.append("\n")
                i += 2
            elif next_c == "t":
                result.append("\t")
                i += 2
            elif next_c == "r":
                result.append("\r")
                i += 2
            elif next_c == "f":
                result.append("\f")
                i += 2
            elif next_c == "v":
                result.append("\v")
                i += 2
            elif next_c == "\\":
                result.append("\\")
                i += 2
            elif next_c == "'":
                result.append("'")
                i += 2
            elif next_c == '"':
                result.append('"')
                i += 2
            elif next_c == "0":
                result.append("\0")
                i += 2
            elif next_c == "x":
                if i + 3 >= len(s):
                    raise ParseError("invalid \\x escape", lineno, col)
                hex_val = s[i + 2 : i + 4]
                try:
                    result.append(chr(int(hex_val, 16)))
                    i += 4
                except ValueError:
                    raise ParseError("invalid \\x escape", lineno, col)
            elif next_c == "u" and not is_bytes:
                if i + 5 >= len(s):
                    raise ParseError("invalid \\u escape", lineno, col)
                hex_val = s[i + 2 : i + 6]
                try:
                    result.append(chr(int(hex_val, 16)))
                    i += 6
                except ValueError:
                    raise ParseError("invalid \\u escape", lineno, col)
            elif next_c == "U" and not is_bytes:
                if i + 9 >= len(s):
                    raise ParseError("invalid \\U escape", lineno, col)
                hex_val = s[i + 2 : i + 10]
                try:
                    code_point = int(hex_val, 16)
                    if code_point > 0x10FFFF:
                        raise ParseError("invalid \\U escape", lineno, col)
                    result.append(chr(code_point))
                    i += 10
                except ValueError:
                    raise ParseError("invalid \\U escape", lineno, col)
            elif next_c == "N" and not is_bytes:
                if i + 2 >= len(s) or s[i + 2] != "{":
                    raise ParseError("invalid \\N escape", lineno, col)
                close_brace = s.find("}", i + 3)
                if close_brace == -1:
                    raise ParseError("invalid \\N escape", lineno, col)
                name = s[i + 3 : close_brace]
                if len(name) == 0:
                    raise ParseError("invalid \\N escape: empty name", lineno, col)
                # Accept the name without resolving — would need unicodedata
                result.append("\\N{" + name + "}")
                i = close_brace + 1
            elif next_c == "\n":
                # Line continuation
                i += 2
            else:
                result.append(c)
                i += 1
        else:
            result.append(c)
            i += 1

    combined = "".join(result)
    if is_bytes:
        return combined.encode("latin-1")
    return combined


def _fstring_find_expr_end(
    content: str, start: int, lineno: int, col: int
) -> tuple[str, str, str, int]:
    """Extract expression, conversion, and format_spec from f-string {expr}.
    Returns (expr_str, conversion, format_spec_str, end_pos).
    end_pos points past the closing }.
    """
    i = start
    length = len(content)
    depth = 1
    expr_parts: list[str] = []
    conversion = ""
    format_spec = ""
    in_format = False
    format_depth = 0

    while i < length and depth > 0:
        ch = content[i]
        if ch == "\\":
            raise ParseError("f-string expression part cannot include a backslash", lineno, col)
        if ch in "\"'":
            # Skip string literal inside expression
            quote = ch
            triple = False
            if i + 2 < length and content[i + 1] == quote and content[i + 2] == quote:
                triple = True
            if triple:
                end_q = content.find(quote + quote + quote, i + 3)
                if end_q == -1:
                    raise ParseError("unterminated string in f-string expression", lineno, col)
                substr = content[i : end_q + 3]
                if not in_format:
                    expr_parts.append(substr)
                else:
                    format_spec = format_spec + substr
                i = end_q + 3
                continue
            else:
                j = i + 1
                while j < length and content[j] != quote:
                    if content[j] == "\\":
                        raise ParseError(
                            "f-string expression part cannot include a backslash", lineno, col
                        )
                    j += 1
                if j >= length:
                    raise ParseError("unterminated string in f-string expression", lineno, col)
                substr = content[i : j + 1]
                if not in_format:
                    expr_parts.append(substr)
                else:
                    format_spec = format_spec + substr
                i = j + 1
                continue
        if ch == "#":
            raise ParseError("f-string expression part cannot include '#'", lineno, col)
        if not in_format:
            if ch == "{":
                depth += 1
                expr_parts.append(ch)
                i += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
                expr_parts.append(ch)
                i += 1
                continue
            if ch == "!" and depth == 1:
                # Conversion specifier - peek ahead
                if i + 1 < length and content[i + 1] in "sra":
                    if i + 2 < length and content[i + 2] in ":}":
                        conversion = content[i + 1]
                        i += 2
                        if i < length and content[i] == ":":
                            in_format = True
                            format_depth = depth
                            i += 1
                        elif i < length and content[i] == "}":
                            depth -= 1
                            i += 1
                            break
                        continue
                    elif i + 2 >= length:
                        conversion = content[i + 1]
                        i += 2
                        continue
                # Invalid conversion: !{ or !<invalid>
                if i + 1 < length and content[i + 1] == "{":
                    raise ParseError("f-string: expecting '}'", lineno, col)
                if i + 1 < length and content[i + 1] not in "sra":
                    if i + 1 < length and content[i + 1] == "}":
                        raise ParseError(
                            "f-string: missing conversion character", lineno, col
                        )
                    raise ParseError(
                        "f-string: invalid conversion character", lineno, col
                    )
                expr_parts.append(ch)
                i += 1
                continue
            if ch == "=" and depth == 1:
                # Debug format: {expr=} or {expr=!s} or {expr=:fmt}
                if i + 1 < length and content[i + 1] == "!":
                    # {expr=!conv}
                    if i + 2 < length and content[i + 2] in "sra":
                        if i + 3 < length and content[i + 3] in ":}":
                            conversion = content[i + 2]
                            i += 3
                            if i < length and content[i] == ":":
                                in_format = True
                                format_depth = depth
                                i += 1
                            elif i < length and content[i] == "}":
                                depth -= 1
                                i += 1
                                break
                            continue
                    # Invalid: {expr=!{...}} or {expr=!b}
                    if i + 2 < length and content[i + 2] == "{":
                        raise ParseError("f-string: expecting '}'", lineno, col)
                    raise ParseError(
                        "f-string: invalid conversion character", lineno, col
                    )
                if i + 1 < length and content[i + 1] == ":":
                    # {expr=:fmt}
                    i += 2
                    in_format = True
                    format_depth = depth
                    continue
                if i + 1 < length and content[i + 1] == "}":
                    # {expr=}
                    i += 2
                    depth -= 1
                    break
                # Just an = inside expression (e.g. == comparison)
                expr_parts.append(ch)
                i += 1
                continue
            if ch == ":" and depth == 1:
                in_format = True
                format_depth = depth
                i += 1
                continue
            expr_parts.append(ch)
            i += 1
        else:
            # Inside format spec
            if ch == "{":
                depth += 1
                format_spec = format_spec + ch
                i += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
                format_spec = format_spec + ch
                i += 1
            else:
                format_spec = format_spec + ch
                i += 1

    expr_str = "".join(expr_parts)
    return (expr_str, conversion, format_spec, i)


def parse_fstring(token_value: str, lineno: int, col: int) -> list[ASTNode]:
    """Parse f-string token to list of Constant and FormattedValue nodes."""
    prefix_end = 0
    while prefix_end < len(token_value) and token_value[prefix_end] not in "\"'":
        prefix_end += 1
    prefix = token_value[:prefix_end].lower()
    is_raw = "r" in prefix
    if token_value[prefix_end : prefix_end + 3] in ('"""', "'''"):
        content = token_value[prefix_end + 3 : -3]
    else:
        content = token_value[prefix_end + 1 : -1]
    values: list[ASTNode] = []
    i = 0
    current_str = ""
    while i < len(content):
        c = content[i]
        if c == "{" and i + 1 < len(content) and content[i + 1] == "{":
            current_str = current_str + "{"
            i += 2
            continue
        if c == "}" and i + 1 < len(content) and content[i + 1] == "}":
            current_str = current_str + "}"
            i += 2
            continue
        if c == "}":
            raise ParseError("f-string: single '}' is not allowed", lineno, col)
        if c == "{":
            if len(current_str) > 0:
                if is_raw:
                    values.append(make_node("Constant", lineno, col, {"value": current_str}))
                else:
                    processed = process_escapes(current_str, False, lineno, col)
                    values.append(make_node("Constant", lineno, col, {"value": processed}))
                current_str = ""
            expr_str, conversion, format_spec_str, new_i = _fstring_find_expr_end(
                content, i + 1, lineno, col
            )
            expr_str = expr_str.strip()
            if len(expr_str) == 0:
                raise ParseError("f-string: empty expression not allowed", lineno, col)
            # Check for semicolons in expression
            if ";" in expr_str:
                raise ParseError(
                    "f-string expression part cannot include ';'", lineno, col
                )
            expr_node = parse_fstring_expr(expr_str, lineno, col)
            conv_int = -1
            if conversion == "s":
                conv_int = ord("s")
            elif conversion == "r":
                conv_int = ord("r")
            elif conversion == "a":
                conv_int = ord("a")
            fmt_spec: ASTNode | None = None
            if len(format_spec_str) > 0:
                # Parse format spec as nested f-string content
                fmt_values = parse_fstring(
                    "f'" + format_spec_str + "'", lineno, col
                )
                if len(fmt_values) > 0:
                    fmt_spec = make_node(
                        "JoinedStr", lineno, col, {"values": fmt_values}
                    )
            fmt_value = make_node(
                "FormattedValue",
                lineno,
                col,
                {"value": expr_node, "conversion": conv_int, "format_spec": fmt_spec},
            )
            values.append(fmt_value)
            i = new_i
            continue
        current_str = current_str + c
        i += 1
    if len(current_str) > 0:
        if is_raw:
            values.append(make_node("Constant", lineno, col, {"value": current_str}))
        else:
            processed = process_escapes(current_str, False, lineno, col)
            values.append(make_node("Constant", lineno, col, {"value": processed}))
    return values


def parse_fstring_expr(expr_str: str, lineno: int, col: int) -> ASTNode:
    """Parse expression inside f-string {expr}."""
    tokens = tokenize(expr_str)
    parser = Parser(tokens)
    result = parser.parse_testlist_star_expr()
    if isinstance(result, dict) and result.get("_type") == "Starred":
        raise ParseError("f-string: starred expression is not allowed here", lineno, col)
    return result


def augassign_op(op_str: str) -> ASTNode:
    """Convert augmented assignment operator to AST op node."""
    ops: dict[str, str] = {
        "+=": "Add",
        "-=": "Sub",
        "*=": "Mult",
        "/=": "Div",
        "//=": "FloorDiv",
        "%=": "Mod",
        "**=": "Pow",
        "@=": "MatMult",
        "&=": "BitAnd",
        "|=": "BitOr",
        "^=": "BitXor",
        ">>=": "RShift",
        "<<=": "LShift",
    }
    type_name = ops.get(op_str)
    if type_name is None:
        type_name = "Add"
    return {"_type": type_name}


def _collect_names(node: ASTNode, names: set[str]) -> None:
    """Collect Name.id values from an assignment target."""
    if not isinstance(node, dict):
        return
    ntype = node.get("_type")
    if ntype == "Name":
        nid = node.get("id")
        if isinstance(nid, str):
            names.add(nid)
    elif ntype in ("Tuple", "List"):
        elts = node.get("elts")
        if isinstance(elts, list):
            i = 0
            while i < len(elts):
                _collect_names(elts[i], names)
                i += 1
    elif ntype == "Starred":
        value = node.get("value")
        if isinstance(value, dict):
            _collect_names(value, names)


def _find_named_exprs(node: ASTNode, results: list[ASTNode]) -> None:
    """Walk AST node to find all NamedExpr nodes."""
    if not isinstance(node, dict):
        return
    if node.get("_type") == "NamedExpr":
        results.append(node)
    for v in node.values():
        if isinstance(v, dict):
            _find_named_exprs(v, results)
        elif isinstance(v, list):
            i = 0
            while i < len(v):
                item = v[i]
                if isinstance(item, dict):
                    _find_named_exprs(item, results)
                i += 1


def _check_comp_walrus(
    elts: list[ASTNode],
    generators: list[ASTNode],
    in_class_comp: bool,
) -> None:
    """Validate walrus operators in comprehension elements and conditions."""
    target_names: set[str] = set()
    i = 0
    while i < len(generators):
        gen = generators[i]
        if isinstance(gen, dict):
            target = gen.get("target")
            if isinstance(target, dict):
                _collect_names(target, target_names)
        i += 1
    named_exprs: list[ASTNode] = []
    i = 0
    while i < len(elts):
        _find_named_exprs(elts[i], named_exprs)
        i += 1
    i = 0
    while i < len(generators):
        gen = generators[i]
        if isinstance(gen, dict):
            ifs = gen.get("ifs")
            if isinstance(ifs, list):
                j = 0
                while j < len(ifs):
                    _find_named_exprs(ifs[j], named_exprs)
                    j += 1
        i += 1
    i = 0
    while i < len(named_exprs):
        ne = named_exprs[i]
        if in_class_comp:
            lineno = ne.get("lineno", 1)
            col = ne.get("col_offset", 0)
            raise ParseError(
                "assignment expression within a comprehension cannot be used in a class body",
                lineno,
                col,
            )
        target = ne.get("target")
        if isinstance(target, dict) and target.get("_type") == "Name":
            name = target.get("id")
            if isinstance(name, str) and name in target_names:
                lineno = ne.get("lineno", 1)
                col = ne.get("col_offset", 0)
                raise ParseError(
                    "assignment expression cannot rebind comprehension iteration variable '"
                    + name
                    + "'",
                    lineno,
                    col,
                )
        i += 1


def _check_async_generator_return(body: list[ASTNode], func_tok: Token) -> None:
    """Check that an async generator doesn't have 'return value'."""
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, dict):
            stype = stmt.get("_type")
            if stype == "Return" and stmt.get("value") is not None:
                lineno = stmt.get("lineno", func_tok.lineno)
                col = stmt.get("col_offset", func_tok.col)
                if not isinstance(lineno, int):
                    lineno = func_tok.lineno
                if not isinstance(col, int):
                    col = func_tok.col
                raise ParseError("'return' with value in async generator", lineno, col)
            if stype in ("If", "While"):
                _check_async_generator_return_list(stmt.get("body"), func_tok)
                _check_async_generator_return_list(stmt.get("orelse"), func_tok)
            elif stype in ("For", "AsyncFor"):
                _check_async_generator_return_list(stmt.get("body"), func_tok)
                _check_async_generator_return_list(stmt.get("orelse"), func_tok)
            elif stype in ("Try", "TryStar"):
                _check_async_generator_return_list(stmt.get("body"), func_tok)
                _check_async_generator_return_list(stmt.get("orelse"), func_tok)
                _check_async_generator_return_list(stmt.get("finalbody"), func_tok)
                handlers = stmt.get("handlers")
                if isinstance(handlers, list):
                    j = 0
                    while j < len(handlers):
                        h = handlers[j]
                        if isinstance(h, dict):
                            _check_async_generator_return_list(h.get("body"), func_tok)
                        j += 1
            elif stype in ("With", "AsyncWith"):
                _check_async_generator_return_list(stmt.get("body"), func_tok)
        i += 1


def _check_async_generator_return_list(
    body: object, func_tok: Token
) -> None:
    if isinstance(body, list):
        _check_async_generator_return(body, func_tok)


INVALID_TARGET_TYPES: set[str] = {
    "Constant",
    "BoolOp",
    "BinOp",
    "UnaryOp",
    "Compare",
    "Lambda",
    "IfExp",
    "ListComp",
    "SetComp",
    "DictComp",
    "GeneratorExp",
    "Call",
    "JoinedStr",
    "FormattedValue",
    "Await",
    "Yield",
    "YieldFrom",
    "Dict",
    "Set",
}


def _node_error(node: ASTNode, msg: str) -> ParseError:
    """Create ParseError from an AST node's position."""
    lineno = node.get("lineno", 1)
    col = node.get("col_offset", 0)
    if not isinstance(lineno, int):
        lineno = 1
    if not isinstance(col, int):
        col = 0
    return ParseError(msg, lineno, col)


def _is_debug_name(node: ASTNode) -> bool:
    return node.get("_type") == "Name" and node.get("id") == "__debug__"


def validate_target(
    node: ASTNode,
    ctx: str,
    is_augassign: bool,
    is_namedexpr: bool,
    is_annotation: bool,
) -> None:
    """Validate that node is a legal assignment/del target."""
    if not isinstance(node, dict):
        return
    node_type = node.get("_type")
    if node_type is None:
        return
    if _is_debug_name(node):
        raise _node_error(node, "cannot assign to __debug__")
    if is_namedexpr:
        if node_type in INVALID_TARGET_TYPES:
            raise _node_error(node, "cannot use assignment expression with " + str(node_type))
        if node_type in ("Attribute", "Subscript"):
            raise _node_error(node, "cannot use assignment expression with " + str(node_type))
        if node_type in ("Starred", "Tuple", "List", "Set", "Dict"):
            raise _node_error(node, "cannot use assignment expression with " + str(node_type))
        return
    if is_annotation:
        if node_type not in ("Name", "Attribute", "Subscript"):
            raise _node_error(node, "only single target (not " + str(node_type) + ") can be annotated")
        return
    if is_augassign:
        if node_type not in ("Name", "Attribute", "Subscript"):
            raise _node_error(node, "illegal expression for augmented assignment")
        return
    if node_type in INVALID_TARGET_TYPES:
        raise _node_error(node, "cannot assign to " + str(node_type))
    if ctx == "Del":
        if node_type == "Starred":
            raise _node_error(node, "cannot use starred expression in del")
    if node_type in ("Tuple", "List"):
        elts = node.get("elts")
        if elts is not None and isinstance(elts, list):
            i = 0
            while i < len(elts):
                validate_target(elts[i], ctx, False, False, False)
                i += 1
    elif node_type == "Starred":
        value = node.get("value")
        if value is not None and isinstance(value, dict):
            validate_target(value, ctx, False, False, False)


def set_context(node: ASTNode, ctx_name: str) -> None:
    """Set the context of a node (Load, Store, Del)."""
    if not isinstance(node, dict):
        return
    validate_target(node, ctx_name, False, False, False)
    if "ctx" in node:
        node["ctx"] = {"_type": ctx_name}
    node_type = node.get("_type")
    if node_type == "Tuple":
        elts = node.get("elts")
        if elts is not None and isinstance(elts, list):
            i = 0
            while i < len(elts):
                set_context(elts[i], ctx_name)
                i += 1
    elif node_type == "List":
        elts = node.get("elts")
        if elts is not None and isinstance(elts, list):
            i = 0
            while i < len(elts):
                set_context(elts[i], ctx_name)
                i += 1
    elif node_type == "Starred":
        value = node.get("value")
        if value is not None and isinstance(value, dict):
            set_context(value, ctx_name)


def set_context_list(nodes: list[ASTNode], ctx_name: str) -> None:
    """Set context on a list of nodes."""
    i = 0
    while i < len(nodes):
        set_context(nodes[i], ctx_name)
        i += 1


def parse(source: str) -> ASTNode:
    """Parse Python source to dict-based AST."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_module()
