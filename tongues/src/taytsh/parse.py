"""Taytsh parser — recursive descent, one method per grammar production."""

from __future__ import annotations

from .ast import (
    Pos,
    TArg,
    TAssignStmt,
    TBinaryOp,
    TBoolLit,
    TBreakStmt,
    TByteLit,
    TBytesLit,
    TCall,
    TCatch,
    TContinueStmt,
    TDecl,
    TDefault,
    TEnumDecl,
    TExpr,
    TExprStmt,
    TFieldAccess,
    TFieldDecl,
    TFnDecl,
    TFnLit,
    TFloatLit,
    TForStmt,
    TFuncType,
    TIdentType,
    TIfStmt,
    TIndex,
    TIntLit,
    TInterfaceDecl,
    TLetStmt,
    TListLit,
    TListType,
    TMapLit,
    TMapType,
    TMatchCase,
    TMatchStmt,
    TModule,
    TNilLit,
    TOpAssignStmt,
    TOptionalType,
    TParam,
    TPatternEnum,
    TPatternNil,
    TPatternType,
    TPrimitive,
    TRange,
    TReturnStmt,
    TRuneLit,
    TSetLit,
    TSetType,
    TSlice,
    TStmt,
    TStringLit,
    TStructDecl,
    TThrowStmt,
    TTernary,
    TTupleAccess,
    TTupleAssignStmt,
    TTupleLit,
    TTupleType,
    TTryStmt,
    TType,
    TUnaryOp,
    TUnionType,
    TVar,
    TWhileStmt,
)
from .tokens import (
    TK_BYTE,
    TK_BYTES,
    TK_EOF,
    TK_FLOAT,
    TK_IDENT,
    TK_INT,
    TK_OP,
    TK_RUNE,
    TK_STRING,
    Token,
)

ASSIGN_OPS: set[str] = {
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<<=",
    ">>=",
}

COMPARE_OPS: set[str] = {"==", "!=", "<", "<=", ">", ">="}

PRIMITIVE_TYPES: set[str] = {
    "int",
    "float",
    "bool",
    "byte",
    "bytes",
    "string",
    "rune",
    "void",
    "obj",
    "nil",
}


class ParseError(Exception):
    """Parse error with location info."""

    def __init__(self, msg: str, line: int, col: int):
        self.msg: str = msg
        self.line: int = line
        self.col: int = col
        super().__init__(msg + " at line " + str(line) + " col " + str(col))


class Parser:
    """Recursive descent parser for Taytsh."""

    def __init__(self, tokens: list[Token]):
        self.tokens: list[Token] = tokens
        self.pos: int = 0

    # ── Helpers ──────────────────────────────────────────────

    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int) -> Token:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return self.tokens[len(self.tokens) - 1]
        return self.tokens[idx]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def at(self, value: str) -> bool:
        return self.current().value == value

    def at_type(self, type_: str) -> bool:
        return self.current().type == type_

    def at_ident(self) -> bool:
        return self.current().type == TK_IDENT

    def expect(self, value: str) -> Token:
        tok = self.current()
        if tok.value != value:
            raise self.error("expected '" + value + "', got '" + tok.value + "'")
        return self.advance()

    def expect_ident(self) -> Token:
        tok = self.current()
        if tok.type != TK_IDENT:
            raise self.error("expected identifier, got '" + tok.value + "'")
        return self.advance()

    def error(self, msg: str) -> ParseError:
        tok = self.current()
        return ParseError(msg, tok.line, tok.col)

    def _pos(self) -> Pos:
        tok = self.current()
        return Pos(tok.line, tok.col)

    def _tok_pos(self, tok: Token) -> Pos:
        return Pos(tok.line, tok.col)

    # ── Top Level ────────────────────────────────────────────

    def parse_program(self) -> TModule:
        decls: list[TDecl] = []
        while not self.at_type(TK_EOF):
            decls.append(self.parse_decl())
        return TModule(decls)

    def parse_decl(self) -> TDecl:
        if self.at("fn"):
            return self.parse_fn_decl()
        if self.at("struct"):
            return self.parse_struct_decl()
        if self.at("interface"):
            return self.parse_interface_decl()
        if self.at("enum"):
            return self.parse_enum_decl()
        raise self.error("expected declaration (fn, struct, interface, enum)")

    def parse_fn_decl(self) -> TFnDecl:
        pos = self._pos()
        self.expect("fn")
        name_tok = self.expect_ident()
        self.expect("(")
        params = self.parse_param_list()
        self.expect(")")
        self.expect("->")
        ret = self.parse_type()
        body = self.parse_block()
        return TFnDecl(pos, name_tok.value, params, ret, body)

    def parse_param_list(self) -> list[TParam]:
        params: list[TParam] = []
        if self.at(")"):
            return params
        if self.at("self"):
            pos = self._pos()
            self.advance()
            params.append(TParam(pos, "self", None))
            while self.at(","):
                self.advance()
                params.append(self.parse_param())
            return params
        params.append(self.parse_param())
        while self.at(","):
            self.advance()
            params.append(self.parse_param())
        return params

    def parse_param(self) -> TParam:
        pos = self._pos()
        name_tok = self.expect_ident()
        self.expect(":")
        typ = self.parse_type()
        return TParam(pos, name_tok.value, typ)

    def parse_block(self) -> list[TStmt]:
        self.expect("{")
        stmts: list[TStmt] = []
        while not self.at("}"):
            stmts.append(self.parse_stmt())
        self.expect("}")
        return stmts

    def parse_struct_decl(self) -> TStructDecl:
        pos = self._pos()
        self.expect("struct")
        name_tok = self.expect_ident()
        parent: str | None = None
        if self.at(":"):
            self.advance()
            parent_tok = self.expect_ident()
            parent = parent_tok.value
        self.expect("{")
        methods: list[TFnDecl] = []
        fields: list[TFieldDecl] = []
        # StructBody = FnDecl* FieldDecl ( FieldDecl | FnDecl )*
        # Leading methods before first field
        while self.at("fn"):
            methods.append(self.parse_fn_decl())
        # At least one field required (grammar says FieldDecl after FnDecl*)
        if self.at("}"):
            raise self.error("struct must have at least one field")
        fields.append(self.parse_field_decl())
        while not self.at("}"):
            if self.at("fn"):
                methods.append(self.parse_fn_decl())
            else:
                fields.append(self.parse_field_decl())
        self.expect("}")
        return TStructDecl(pos, name_tok.value, parent, fields, methods)

    def parse_field_decl(self) -> TFieldDecl:
        pos = self._pos()
        name_tok = self.expect_ident()
        self.expect(":")
        typ = self.parse_type()
        return TFieldDecl(pos, name_tok.value, typ)

    def parse_interface_decl(self) -> TInterfaceDecl:
        pos = self._pos()
        self.expect("interface")
        name_tok = self.expect_ident()
        self.expect("{")
        self.expect("}")
        return TInterfaceDecl(pos, name_tok.value)

    def parse_enum_decl(self) -> TEnumDecl:
        pos = self._pos()
        self.expect("enum")
        name_tok = self.expect_ident()
        self.expect("{")
        variants: list[str] = []
        while self.at_ident():
            variants.append(self.advance().value)
        if len(variants) == 0:
            raise self.error("enum must have at least one variant")
        self.expect("}")
        return TEnumDecl(pos, name_tok.value, variants)

    # ── Types ────────────────────────────────────────────────

    def parse_type(self) -> TType:
        """Type = UnionType ( '?' )?"""
        typ = self.parse_union_type()
        if self.at("?"):
            self.advance()
            return TOptionalType(typ.pos, typ)
        return typ

    def parse_union_type(self) -> TType:
        """UnionType = BaseType ( '|' BaseType )*"""
        first = self.parse_base_type()
        if not self.at("|"):
            return first
        members: list[TType] = [first]
        while self.at("|"):
            self.advance()
            members.append(self.parse_base_type())
        return TUnionType(first.pos, members)

    def parse_base_type(self) -> TType:
        """Parse a single base type."""
        pos = self._pos()
        tok = self.current()
        if tok.value in PRIMITIVE_TYPES:
            self.advance()
            return TPrimitive(pos, tok.value)
        if tok.value == "list":
            self.advance()
            self.expect("[")
            elem = self.parse_type()
            self.expect("]")
            return TListType(pos, elem)
        if tok.value == "map":
            self.advance()
            self.expect("[")
            key = self.parse_type()
            self.expect(",")
            val = self.parse_type()
            self.expect("]")
            return TMapType(pos, key, val)
        if tok.value == "set":
            self.advance()
            self.expect("[")
            elem = self.parse_type()
            self.expect("]")
            return TSetType(pos, elem)
        if tok.value == "fn":
            self.advance()
            self.expect("[")
            params: list[TType] = [self.parse_type()]
            while self.at(","):
                self.advance()
                params.append(self.parse_type())
            self.expect("]")
            return TFuncType(pos, params)
        if tok.value == "(":
            self.advance()
            first = self.parse_type()
            self.expect(",")
            elements: list[TType] = [first, self.parse_type()]
            while self.at(","):
                self.advance()
                elements.append(self.parse_type())
            self.expect(")")
            return TTupleType(pos, elements)
        if tok.type == TK_IDENT:
            self.advance()
            return TIdentType(pos, tok.value)
        raise self.error("expected type, got '" + tok.value + "'")

    def parse_type_name(self) -> TType:
        """TypeName for catch/case — same alternatives as BaseType."""
        return self.parse_base_type()

    # ── Statements ───────────────────────────────────────────

    def parse_stmt(self) -> TStmt:
        tok = self.current()
        if tok.value == "let":
            return self.parse_let_stmt()
        if tok.value == "if":
            return self.parse_if_stmt()
        if tok.value == "while":
            return self.parse_while_stmt()
        if tok.value == "for":
            return self.parse_for_stmt()
        if tok.value == "match":
            return self.parse_match_stmt()
        if tok.value == "try":
            return self.parse_try_stmt()
        if tok.value == "return":
            return self.parse_return_stmt()
        if tok.value == "break":
            pos = self._pos()
            self.advance()
            return TBreakStmt(pos)
        if tok.value == "continue":
            pos = self._pos()
            self.advance()
            return TContinueStmt(pos)
        if tok.value == "throw":
            return self.parse_throw_stmt()
        return self.parse_expr_stmt()

    def parse_let_stmt(self) -> TLetStmt:
        pos = self._pos()
        self.expect("let")
        name_tok = self.expect_ident()
        self.expect(":")
        typ = self.parse_type()
        value: TExpr | None = None
        if self.at("="):
            self.advance()
            value = self.parse_expr()
        return TLetStmt(pos, name_tok.value, typ, value)

    def parse_if_stmt(self) -> TIfStmt:
        pos = self._pos()
        self.expect("if")
        cond = self.parse_expr()
        then_body = self.parse_block()
        else_body: list[TStmt] | None = None
        if self.at("else"):
            self.advance()
            if self.at("if"):
                else_body = [self.parse_if_stmt()]
            else:
                else_body = self.parse_block()
        return TIfStmt(pos, cond, then_body, else_body)

    def parse_while_stmt(self) -> TWhileStmt:
        pos = self._pos()
        self.expect("while")
        cond = self.parse_expr()
        body = self.parse_block()
        return TWhileStmt(pos, cond, body)

    def parse_for_stmt(self) -> TForStmt:
        pos = self._pos()
        self.expect("for")
        first_name = self.expect_ident()
        binding: list[str] = [first_name.value]
        if self.at(","):
            self.advance()
            second_name = self.expect_ident()
            binding.append(second_name.value)
        self.expect("in")
        iterable: TExpr | TRange
        if self.at("range"):
            iterable = self.parse_range()
        else:
            iterable = self.parse_expr()
        body = self.parse_block()
        return TForStmt(pos, binding, iterable, body)

    def parse_range(self) -> TRange:
        pos = self._pos()
        self.expect("range")
        self.expect("(")
        args: list[TExpr] = [self.parse_expr()]
        while self.at(","):
            self.advance()
            args.append(self.parse_expr())
        self.expect(")")
        return TRange(pos, args)

    def parse_match_stmt(self) -> TMatchStmt:
        pos = self._pos()
        self.expect("match")
        expr = self.parse_expr()
        self.expect("{")
        cases: list[TMatchCase] = []
        default: TDefault | None = None
        while self.at("case"):
            cases.append(self.parse_case())
        if self.at("default"):
            default = self.parse_default()
        self.expect("}")
        if len(cases) == 0 and default is None:
            raise ParseError(
                "match must have at least one case or default", pos.line, pos.col
            )
        return TMatchStmt(pos, expr, cases, default)

    def parse_case(self) -> TMatchCase:
        pos = self._pos()
        self.expect("case")
        pattern = self.parse_pattern()
        body = self.parse_block()
        return TMatchCase(pos, pattern, body)

    def parse_pattern(self) -> TPatternType | TPatternEnum | TPatternNil:
        pos = self._pos()
        if self.at("nil"):
            self.advance()
            return TPatternNil(pos)
        first = self.expect_ident()
        if self.at(":"):
            self.advance()
            type_name = self.parse_type_name()
            return TPatternType(pos, first.value, type_name)
        if self.at("."):
            self.advance()
            variant = self.expect_ident()
            return TPatternEnum(pos, first.value, variant.value)
        raise self.error("expected ':' or '.' in case pattern")

    def parse_default(self) -> TDefault:
        pos = self._pos()
        self.expect("default")
        name: str | None = None
        if self.at_ident():
            name_tok = self.advance()
            self.expect(":")
            self.expect("obj")
            name = name_tok.value
        body = self.parse_block()
        return TDefault(pos, name, body)

    def parse_try_stmt(self) -> TTryStmt:
        pos = self._pos()
        self.expect("try")
        body = self.parse_block()
        catches: list[TCatch] = []
        finally_body: list[TStmt] | None = None
        while self.at("catch"):
            catches.append(self.parse_catch())
        if self.at("finally"):
            self.advance()
            finally_body = self.parse_block()
        if len(catches) == 0 and finally_body is None:
            raise ParseError("try must have catch or finally", pos.line, pos.col)
        return TTryStmt(pos, body, catches, finally_body)

    def parse_catch(self) -> TCatch:
        pos = self._pos()
        self.expect("catch")
        name_tok = self.expect_ident()
        self.expect(":")
        types: list[TType] = [self.parse_type_name()]
        while self.at("|"):
            self.advance()
            types.append(self.parse_type_name())
        body = self.parse_block()
        return TCatch(pos, name_tok.value, types, body)

    def parse_return_stmt(self) -> TReturnStmt:
        pos = self._pos()
        self.expect("return")
        value: TExpr | None = None
        if self._at_expr_start():
            value = self.parse_expr()
        return TReturnStmt(pos, value)

    def parse_throw_stmt(self) -> TThrowStmt:
        pos = self._pos()
        self.expect("throw")
        expr = self.parse_expr()
        return TThrowStmt(pos, expr)

    def parse_expr_stmt(self) -> TStmt:
        """ExprStmt = Expr ( AssignTail )?"""
        pos = self._pos()
        expr = self.parse_expr()
        tok = self.current()
        if tok.value == "=" and tok.type == TK_OP:
            self.advance()
            value = self.parse_expr()
            return TAssignStmt(pos, expr, value)
        if tok.value in ASSIGN_OPS and tok.value != "=":
            op = tok.value
            self.advance()
            value = self.parse_expr()
            return TOpAssignStmt(pos, expr, op, value)
        if tok.value == ",":
            targets: list[TExpr] = [expr]
            while self.at(","):
                self.advance()
                targets.append(self.parse_expr())
            self.expect("=")
            value = self.parse_expr()
            return TTupleAssignStmt(pos, targets, value)
        return TExprStmt(pos, expr)

    def _at_expr_start(self) -> bool:
        """Check if current token can start an expression."""
        tok = self.current()
        if tok.type in (
            TK_INT,
            TK_FLOAT,
            TK_BYTE,
            TK_STRING,
            TK_RUNE,
            TK_BYTES,
            TK_IDENT,
        ):
            return True
        if tok.value in ("true", "false", "nil", "self"):
            return True
        if tok.value in ("(", "[", "{"):
            return True
        if tok.value in ("-", "!", "~"):
            return True
        return False

    # ── Expressions ──────────────────────────────────────────

    def parse_expr(self) -> TExpr:
        return self.parse_ternary()

    def parse_ternary(self) -> TExpr:
        """Ternary = Or ( '?' Expr ':' Ternary )?"""
        expr = self.parse_or()
        if self.at("?"):
            self.advance()
            then_expr = self.parse_expr()
            self.expect(":")
            else_expr = self.parse_ternary()
            return TTernary(expr.pos, expr, then_expr, else_expr)
        return expr

    def parse_or(self) -> TExpr:
        """Or = And ( '||' And )*"""
        left = self.parse_and()
        while self.at("||"):
            self.advance()
            right = self.parse_and()
            left = TBinaryOp(left.pos, "||", left, right)
        return left

    def parse_and(self) -> TExpr:
        """And = Compare ( '&&' Compare )*"""
        left = self.parse_compare()
        while self.at("&&"):
            self.advance()
            right = self.parse_compare()
            left = TBinaryOp(left.pos, "&&", left, right)
        return left

    def parse_compare(self) -> TExpr:
        """Compare = BitOr ( CompOp BitOr )?"""
        left = self.parse_bit_or()
        tok = self.current()
        if tok.value in COMPARE_OPS:
            op = tok.value
            self.advance()
            right = self.parse_bit_or()
            return TBinaryOp(left.pos, op, left, right)
        return left

    def parse_bit_or(self) -> TExpr:
        """BitOr = BitXor ( '|' BitXor )*"""
        left = self.parse_bit_xor()
        while self.at("|"):
            self.advance()
            right = self.parse_bit_xor()
            left = TBinaryOp(left.pos, "|", left, right)
        return left

    def parse_bit_xor(self) -> TExpr:
        """BitXor = BitAnd ( '^' BitAnd )*"""
        left = self.parse_bit_and()
        while self.at("^"):
            self.advance()
            right = self.parse_bit_and()
            left = TBinaryOp(left.pos, "^", left, right)
        return left

    def parse_bit_and(self) -> TExpr:
        """BitAnd = Shift ( '&' Shift )*"""
        left = self.parse_shift()
        while self.at("&"):
            self.advance()
            right = self.parse_shift()
            left = TBinaryOp(left.pos, "&", left, right)
        return left

    def parse_shift(self) -> TExpr:
        """Shift = Sum ( ( '<<' | '>>' ) Sum )*"""
        left = self.parse_sum()
        while self.at("<<") or self.at(">>"):
            op = self.advance().value
            right = self.parse_sum()
            left = TBinaryOp(left.pos, op, left, right)
        return left

    def parse_sum(self) -> TExpr:
        """Sum = Product ( ( '+' | '-' ) Product )*"""
        left = self.parse_product()
        while self.at("+") or self.at("-"):
            op = self.advance().value
            right = self.parse_product()
            left = TBinaryOp(left.pos, op, left, right)
        return left

    def parse_product(self) -> TExpr:
        """Product = Unary ( ( '*' | '/' | '%' ) Unary )*"""
        left = self.parse_unary()
        while self.at("*") or self.at("/") or self.at("%"):
            op = self.advance().value
            right = self.parse_unary()
            left = TBinaryOp(left.pos, op, left, right)
        return left

    def parse_unary(self) -> TExpr:
        """Unary = ( '-' | '!' | '~' ) Unary | Postfix"""
        tok = self.current()
        if tok.type == TK_OP and (
            tok.value == "-" or tok.value == "!" or tok.value == "~"
        ):
            pos = self._pos()
            op = self.advance().value
            operand = self.parse_unary()
            return TUnaryOp(pos, op, operand)
        return self.parse_postfix()

    def parse_postfix(self) -> TExpr:
        """Postfix = Primary ( Suffix )*"""
        expr = self.parse_primary()
        while True:
            if self.at("."):
                self.advance()
                tok = self.current()
                if tok.type == TK_INT:
                    self.advance()
                    expr = TTupleAccess(expr.pos, expr, int(tok.value))
                elif tok.type == TK_IDENT:
                    self.advance()
                    expr = TFieldAccess(expr.pos, expr, tok.value)
                else:
                    raise self.error("expected field name or tuple index after '.'")
            elif self.at("["):
                self.advance()
                index = self.parse_expr()
                if self.at(":"):
                    self.advance()
                    high = self.parse_expr()
                    self.expect("]")
                    expr = TSlice(expr.pos, expr, index, high)
                else:
                    self.expect("]")
                    expr = TIndex(expr.pos, expr, index)
            elif self.at("("):
                self.advance()
                args = self.parse_arg_list()
                self.expect(")")
                expr = TCall(expr.pos, expr, args)
            else:
                break
        return expr

    def parse_arg_list(self) -> list[TArg]:
        """ArgList = ( Arg ( ',' Arg )* )?"""
        args: list[TArg] = []
        if self.at(")"):
            return args
        args.append(self.parse_arg())
        while self.at(","):
            self.advance()
            args.append(self.parse_arg())
        return args

    def parse_arg(self) -> TArg:
        """Arg = IDENT ':' Expr | Expr  (2-token lookahead for named)"""
        pos = self._pos()
        if self.at_ident() and self.peek(1).value == ":":
            name_tok = self.advance()
            self.advance()  # skip ':'
            value = self.parse_expr()
            return TArg(pos, name_tok.value, value)
        value = self.parse_expr()
        return TArg(pos, None, value)

    def parse_primary(self) -> TExpr:
        """Parse a primary expression."""
        tok = self.current()
        pos = self._pos()

        # Literals
        if tok.type == TK_INT:
            self.advance()
            return TIntLit(pos, int(tok.value), tok.value)
        if tok.type == TK_FLOAT:
            self.advance()
            return TFloatLit(pos, float(tok.value), tok.value)
        if tok.type == TK_BYTE:
            self.advance()
            return TByteLit(pos, int(tok.value, 16), tok.value)
        if tok.type == TK_STRING:
            self.advance()
            return TStringLit(pos, tok.value)
        if tok.type == TK_RUNE:
            self.advance()
            return TRuneLit(pos, tok.value)
        if tok.type == TK_BYTES:
            self.advance()
            return TBytesLit(pos, tok.bytes_value)

        # Bool/nil keywords
        if tok.value == "true":
            self.advance()
            return TBoolLit(pos, True)
        if tok.value == "false":
            self.advance()
            return TBoolLit(pos, False)
        if tok.value == "nil":
            self.advance()
            return TNilLit(pos)

        # Identifier (self is a keyword but valid in expression position)
        if tok.type == TK_IDENT or tok.value == "self":
            self.advance()
            return TVar(pos, tok.value)

        # ( — fn literal, tuple, or parens
        if tok.value == "(":
            if self._is_fn_literal():
                return self.parse_fn_literal()
            self.advance()  # skip (
            first = self.parse_expr()
            if self.at(","):
                elements: list[TExpr] = [first]
                while self.at(","):
                    self.advance()
                    elements.append(self.parse_expr())
                self.expect(")")
                return TTupleLit(pos, elements)
            self.expect(")")
            return first

        # [ — list literal
        if tok.value == "[":
            self.advance()
            elements_list: list[TExpr] = []
            if not self.at("]"):
                elements_list.append(self.parse_expr())
                while self.at(","):
                    self.advance()
                    elements_list.append(self.parse_expr())
            self.expect("]")
            return TListLit(pos, elements_list)

        # { — map or set literal
        if tok.value == "{":
            self.advance()
            if self.at("}"):
                raise self.error("empty {} is ambiguous — use Map() or Set()")
            first = self.parse_expr()
            if self.at(":"):
                # Map literal
                self.advance()
                first_val = self.parse_expr()
                entries: list[tuple[TExpr, TExpr]] = [(first, first_val)]
                while self.at(","):
                    self.advance()
                    k = self.parse_expr()
                    self.expect(":")
                    v = self.parse_expr()
                    entries.append((k, v))
                self.expect("}")
                return TMapLit(pos, entries)
            else:
                # Set literal
                set_elements: list[TExpr] = [first]
                while self.at(","):
                    self.advance()
                    set_elements.append(self.parse_expr())
                self.expect("}")
                return TSetLit(pos, set_elements)

        raise self.error("expected expression, got '" + tok.value + "'")

    def _is_fn_literal(self) -> bool:
        """Lookahead scan: check if '(' begins a fn literal by finding matching ')' then '->'."""
        depth = 1
        i = self.pos + 1
        num_tokens = len(self.tokens)
        while i < num_tokens:
            tok = self.tokens[i]
            if tok.value == "(" or tok.value == "[":
                depth += 1
            elif tok.value == ")" or tok.value == "]":
                depth -= 1
                if depth == 0:
                    return i + 1 < num_tokens and self.tokens[i + 1].value == "->"
            i += 1
        return False

    def parse_fn_literal(self) -> TFnLit:
        """FnLiteral = '(' ParamList ')' '->' Type ( Block | '=>' Expr )"""
        pos = self._pos()
        self.expect("(")
        params = self.parse_param_list()
        self.expect(")")
        self.expect("->")
        ret = self.parse_type()
        if self.at("{"):
            body: list[TStmt] | TExpr = self.parse_block()
            return TFnLit(pos, params, ret, body)
        self.expect("=>")
        body = self.parse_expr()
        return TFnLit(pos, params, ret, body)
