"""Context objects for frontend lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .ast_compat import ASTNode

if TYPE_CHECKING:
    from .. import ir
    from ..ir import FuncInfo, SymbolTable, Type


class TypeContext:
    """Context for bidirectional type inference (Pierce & Turner style)."""

    def __init__(
        self,
        expected: Type | None = None,
        var_types: dict[str, Type] | None = None,
        return_type: Type | None = None,
        tuple_vars: dict[str, list[str]] | None = None,
        sentinel_ints: set[str] | None = None,
        optional_strings: set[str] | None = None,
        narrowed_vars: set[str] | None = None,
        kind_source_vars: dict[str, str] | None = None,
        union_types: dict[str, list[str]] | None = None,
        list_element_unions: dict[str, list[str]] | None = None,
        narrowed_attr_paths: dict[tuple[str, ...], str] | None = None,
        unified_to_node: set[str] | None = None,
    ) -> None:
        self.expected: Type | None = expected
        self.return_type: Type | None = return_type
        self.var_types: dict[str, Type] = var_types if var_types is not None else {}
        self.tuple_vars: dict[str, list[str]] = tuple_vars if tuple_vars is not None else {}
        self.sentinel_ints: set[str] = sentinel_ints if sentinel_ints is not None else set()
        self.optional_strings: set[str] = optional_strings if optional_strings is not None else set()
        self.narrowed_vars: set[str] = narrowed_vars if narrowed_vars is not None else set()
        self.kind_source_vars: dict[str, str] = kind_source_vars if kind_source_vars is not None else {}
        self.union_types: dict[str, list[str]] = union_types if union_types is not None else {}
        self.list_element_unions: dict[str, list[str]] = list_element_unions if list_element_unions is not None else {}
        self.narrowed_attr_paths: dict[tuple[str, ...], str] = narrowed_attr_paths if narrowed_attr_paths is not None else {}
        self.unified_to_node: set[str] = unified_to_node if unified_to_node is not None else set()


@dataclass
class FrontendContext:
    """Immutable-ish context passed through lowering."""

    symbols: "SymbolTable"
    type_ctx: TypeContext
    current_func_info: "FuncInfo | None"
    current_class_name: str
    node_types: set[str]
    hierarchy_root: str | None
    kind_to_struct: dict[str, str]
    kind_to_class: dict[str, str]
    current_catch_var: str | None
    exception_subclasses: set[str]
    source: str = ""  # Original source for literal format detection


@dataclass
class LoweringDispatch:
    """Callbacks for recursive lowering.

    These callbacks allow extracted lowering functions to call back into
    the Frontend instance for recursive lowering and type inference.

    Why these callbacks must stay (cannot be inlined):
    - lower_expr/lower_expr_as_bool/lower_stmts/lower_lvalue: Recursive calls
    - lower_expr_List: Bidirectional type inference for constructors
    - annotation_to_str: Recursive AST traversal with self reference
    - merge_keyword_args/fill_default_args/etc: Pass lower_expr/infer callbacks
    - set_catch_var: Mutates Frontend instance state
    """

    # Recursive lowering (MUST STAY - recursive)
    lower_expr: Callable[[ASTNode], "ir.Expr"]
    lower_expr_as_bool: Callable[[ASTNode], "ir.Expr"]
    lower_stmts: Callable[[list[ASTNode]], list["ir.Stmt"]]
    lower_lvalue: Callable[[ASTNode], "ir.LValue"]
    # Bidirectional type inference (MUST STAY - uses expected type context)
    lower_expr_List: Callable[[ASTNode, "Type | None"], "ir.Expr"]
    # Helper (MUST STAY - recursive AST traversal)
    annotation_to_str: Callable[[ASTNode | None], str]
    # Argument handling (MUST STAY - pass lower_expr/infer callbacks internally)
    merge_keyword_args: Callable[["Type", str, list, ASTNode], list]
    fill_default_args: Callable[["Type", str, list], list]
    merge_keyword_args_for_func: Callable[["FuncInfo", list, ASTNode], list]
    add_address_of_for_ptr_params: Callable[["Type", str, list, list[ASTNode]], list]
    deref_for_slice_params: Callable[["Type", str, list, list[ASTNode]], list]
    deref_for_func_slice_params: Callable[[str, list, list[ASTNode]], list]
    coerce_sentinel_to_ptr: Callable[["Type", str, list, list], list]
    # Type inference (MUST STAY - needed for narrowing contexts)
    infer_expr_type_from_ast: Callable[[ASTNode], "Type"]
    # Exception handling (MUST STAY - mutates instance state)
    set_catch_var: Callable[[str | None], str | None]
