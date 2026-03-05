"""Control flow graph for forward-flow type analysis.

Builds a lightweight CFG from function body AST and resolves types at each
node via worklist-based forward dataflow.  Enables aliased-condition narrowing,
correct branch merging with combine_types, and loop-body type propagation.

Written in the Tongues subset (no generators, closures, lambdas, getattr).
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import (
    TypeNode,
    ASTNode,
    JNull,
    combine_types,
    get_str,
    get_int,
    get_node,
    get_nodes,
    type_eq,
)


# ---------------------------------------------------------------------------
# Flow node types
# ---------------------------------------------------------------------------


@dataclass
class FlowNode:
    """Base flow node."""

    id: int
    prev: list[int]


@dataclass
class FlowStart(FlowNode):
    """Function entry point."""

    pass


@dataclass
class FlowAssign(FlowNode):
    """Variable assignment."""

    name: str
    lineno: int


@dataclass
class FlowNarrow(FlowNode):
    """True-branch narrowing (isinstance / is None / is not None / const_field)."""

    target: str
    narrow_type: str
    type_name: str
    field_name: str


@dataclass
class FlowWiden(FlowNode):
    """False-branch (inverse of a FlowNarrow)."""

    narrow_id: int


@dataclass
class FlowJoin(FlowNode):
    """Merge point after branches."""

    pass


@dataclass
class FlowLoopHead(FlowNode):
    """Loop header that receives the back-edge."""

    pass


@dataclass
class FlowUnreachable(FlowNode):
    """After return / raise / break / continue."""

    pass


@dataclass
class FlowCondAlias(FlowNode):
    """flag = isinstance(x, T)  — stores predicate for later expansion."""

    alias_name: str
    target: str
    narrow_type: str
    type_name: str
    field_name: str


# ---------------------------------------------------------------------------
# FlowGraph container
# ---------------------------------------------------------------------------


class FlowGraph:
    """Container for flow nodes."""

    def __init__(self) -> None:
        self.nodes: list[FlowNode] = []
        self._succ: dict[int, list[int]] = {}

    def add(self, node: FlowNode) -> int:
        self.nodes.append(node)
        return node.id

    def node_at(self, nid: int) -> FlowNode | None:
        if nid < 0 or nid >= len(self.nodes):
            return None
        return self.nodes[nid]

    def next_id(self) -> int:
        return len(self.nodes)

    def successors(self, nid: int) -> list[int]:
        cached = self._succ.get(nid)
        if cached is not None:
            return cached
        result: list[int] = []
        for node in self.nodes:
            for prev in node.prev:
                if prev == nid:
                    result.append(node.id)
        self._succ[nid] = result
        return result


# ---------------------------------------------------------------------------
# Alias detection helpers
# ---------------------------------------------------------------------------


def _is_isinstance_call(node: ASTNode) -> bool:
    """Check if node is isinstance(x, T)."""
    if not isinstance(node, dict):
        return False
    if get_str(node, "_type") != "Call":
        return False
    func = get_node(node, "func")
    if not func:
        return False
    if get_str(func, "_type") != "Name":
        return False
    return get_str(func, "id") == "isinstance"


def _is_none_compare(node: ASTNode) -> tuple[str, str]:
    """Check if node is `x is None` or `x is not None`.

    Returns (target_name, "is_none" | "is_not_none" | "").
    """
    if not isinstance(node, dict):
        return ("", "")
    if get_str(node, "_type") != "Compare":
        return ("", "")
    left = get_node(node, "left")
    ops = get_nodes(node, "ops")
    comparators = get_nodes(node, "comparators")
    if not left or not ops or not comparators:
        return ("", "")
    comp = comparators[0]
    if get_str(comp, "_type") != "Constant":
        return ("", "")
    cv = comp.get("value")
    if cv is not None and not isinstance(cv, JNull):
        return ("", "")
    if get_str(left, "_type") != "Name":
        return ("", "")
    name = get_str(left, "id")
    if not name:
        return ("", "")
    op_type = get_str(ops[0], "_type")
    if op_type in ("Is", "Eq"):
        return (name, "is_none")
    if op_type in ("IsNot", "NotEq"):
        return (name, "is_not_none")
    return ("", "")


def _is_const_field_compare(node: ASTNode) -> tuple[str, str, str]:
    """Check if node is `obj.field == "value"` or `obj.field != "value"`.

    Returns (obj_name, field_name, value) or ("","","").
    """
    if not isinstance(node, dict):
        return ("", "", "")
    if get_str(node, "_type") != "Compare":
        return ("", "", "")
    left = get_node(node, "left")
    ops = get_nodes(node, "ops")
    comparators = get_nodes(node, "comparators")
    if not left or not ops or not comparators:
        return ("", "", "")
    if get_str(left, "_type") != "Attribute":
        return ("", "", "")
    attr = get_str(left, "attr")
    obj_node = get_node(left, "value")
    if not obj_node or get_str(obj_node, "_type") != "Name":
        return ("", "", "")
    obj_name = get_str(obj_node, "id")
    if not obj_name or not attr:
        return ("", "", "")
    comp = comparators[0]
    if get_str(comp, "_type") != "Constant":
        return ("", "", "")
    from .types import JStr

    comp_v = comp.get("value")
    if not isinstance(comp_v, JStr):
        return ("", "", "")
    op_type = get_str(ops[0], "_type")
    if op_type == "Eq":
        return (obj_name, attr, comp_v.value)
    if op_type == "NotEq":
        return (obj_name, attr, comp_v.value)
    return ("", "", "")


def _isinstance_target_and_type(node: ASTNode) -> tuple[str, str]:
    """Extract (target_name, type_name) from isinstance(x, T).

    Only handles single Name type arg.  Returns ("","") on failure.
    """
    args = get_nodes(node, "args")
    if len(args) < 2:
        return ("", "")
    target = args[0]
    type_arg = args[1]
    if get_str(target, "_type") != "Name":
        return ("", "")
    name = get_str(target, "id")
    if not name:
        return ("", "")
    if get_str(type_arg, "_type") != "Name":
        return ("", "")
    tname = get_str(type_arg, "id")
    if not tname:
        return ("", "")
    return (name, tname)


# ---------------------------------------------------------------------------
# CFG builder
# ---------------------------------------------------------------------------


def build_cfg(body: list[ASTNode]) -> FlowGraph:
    """Build a control flow graph from a function body."""
    graph = FlowGraph()
    start = FlowStart(id=graph.next_id(), prev=[])
    graph.add(start)
    _build_stmts(body, start.id, graph)
    return graph


def _build_stmts(stmts: list[ASTNode], prev_id: int, graph: FlowGraph) -> int:
    """Build CFG nodes for a statement list.  Returns last node id."""
    cur = prev_id
    for stmt in stmts:
        if not isinstance(stmt, dict):
            continue
        cur = _build_stmt(stmt, cur, graph)
        node = graph.node_at(cur)
        if node is not None and isinstance(node, FlowUnreachable):
            return cur
    return cur


def _build_stmt(stmt: ASTNode, prev_id: int, graph: FlowGraph) -> int:
    """Build CFG node(s) for a single statement.  Returns last node id."""
    t = get_str(stmt, "_type")
    if t == "Assign":
        return _build_assign(stmt, prev_id, graph)
    if t == "AnnAssign":
        return _build_ann_assign(stmt, prev_id, graph)
    if t == "AugAssign":
        return _build_aug_assign(stmt, prev_id, graph)
    if t == "If":
        return _build_if(stmt, prev_id, graph)
    if t == "While":
        return _build_while(stmt, prev_id, graph)
    if t == "For":
        return _build_for(stmt, prev_id, graph)
    if t == "Return" or t == "Raise":
        return _build_unreachable(stmt, prev_id, graph)
    if t == "Break" or t == "Continue":
        return _build_unreachable(stmt, prev_id, graph)
    if t == "Assert":
        return _build_assert(stmt, prev_id, graph)
    return prev_id


def _build_assign(stmt: ASTNode, prev_id: int, graph: FlowGraph) -> int:
    """Build flow node for an assignment, detecting condition aliases."""
    targets = get_nodes(stmt, "targets")
    value = get_node(stmt, "value")
    lineno = get_int(stmt, "lineno")
    if len(targets) != 1 or not value:
        return prev_id
    tgt = targets[0]
    if get_str(tgt, "_type") != "Name":
        return prev_id
    name = get_str(tgt, "id")
    if not name:
        return prev_id
    if _is_isinstance_call(value):
        tgt_name, type_name = _isinstance_target_and_type(value)
        if tgt_name and type_name:
            nid = graph.next_id()
            node = FlowCondAlias(
                id=nid,
                prev=[prev_id],
                alias_name=name,
                target=tgt_name,
                narrow_type="isinstance",
                type_name=type_name,
                field_name="",
            )
            graph.add(node)
            return nid
    none_target, none_kind = _is_none_compare(value)
    if none_target and none_kind:
        nid = graph.next_id()
        node = FlowCondAlias(
            id=nid,
            prev=[prev_id],
            alias_name=name,
            target=none_target,
            narrow_type=none_kind,
            type_name="",
            field_name="",
        )
        graph.add(node)
        return nid
    cf_obj, cf_field, cf_value = _is_const_field_compare(value)
    if cf_obj and cf_field and cf_value:
        nid = graph.next_id()
        node = FlowCondAlias(
            id=nid,
            prev=[prev_id],
            alias_name=name,
            target=cf_obj,
            narrow_type="const_field",
            type_name=cf_value,
            field_name=cf_field,
        )
        graph.add(node)
        return nid
    nid = graph.next_id()
    assign = FlowAssign(id=nid, prev=[prev_id], name=name, lineno=lineno)
    graph.add(assign)
    return nid


def _build_ann_assign(stmt: ASTNode, prev_id: int, graph: FlowGraph) -> int:
    """Build flow node for annotated assignment, detecting condition aliases."""
    target = get_node(stmt, "target")
    value = get_node(stmt, "value")
    lineno = get_int(stmt, "lineno")
    if not target:
        return prev_id
    if get_str(target, "_type") != "Name":
        return prev_id
    name = get_str(target, "id")
    if not name or not value:
        return prev_id
    if _is_isinstance_call(value):
        tgt_name, type_name = _isinstance_target_and_type(value)
        if tgt_name and type_name:
            nid = graph.next_id()
            node = FlowCondAlias(
                id=nid,
                prev=[prev_id],
                alias_name=name,
                target=tgt_name,
                narrow_type="isinstance",
                type_name=type_name,
                field_name="",
            )
            graph.add(node)
            return nid
    none_target, none_kind = _is_none_compare(value)
    if none_target and none_kind:
        nid = graph.next_id()
        node = FlowCondAlias(
            id=nid,
            prev=[prev_id],
            alias_name=name,
            target=none_target,
            narrow_type=none_kind,
            type_name="",
            field_name="",
        )
        graph.add(node)
        return nid
    cf_obj, cf_field, cf_value = _is_const_field_compare(value)
    if cf_obj and cf_field and cf_value:
        nid = graph.next_id()
        node = FlowCondAlias(
            id=nid,
            prev=[prev_id],
            alias_name=name,
            target=cf_obj,
            narrow_type="const_field",
            type_name=cf_value,
            field_name=cf_field,
        )
        graph.add(node)
        return nid
    nid = graph.next_id()
    assign = FlowAssign(id=nid, prev=[prev_id], name=name, lineno=lineno)
    graph.add(assign)
    return nid


def _build_aug_assign(stmt: ASTNode, prev_id: int, graph: FlowGraph) -> int:
    """Build flow node for augmented assignment (+=, etc)."""
    target = get_node(stmt, "target")
    lineno = get_int(stmt, "lineno")
    if not target:
        return prev_id
    if get_str(target, "_type") != "Name":
        return prev_id
    name = get_str(target, "id")
    if not name:
        return prev_id
    nid = graph.next_id()
    assign = FlowAssign(id=nid, prev=[prev_id], name=name, lineno=lineno)
    graph.add(assign)
    return nid


def _extract_condition_info(
    test: ASTNode,
) -> tuple[str, str, str, str]:
    """Extract narrowing info from a condition expression.

    Returns (target, narrow_type, type_name, field_name) or all empty on failure.
    narrow_type is one of: isinstance, is_none, is_not_none, const_field
    """
    if not isinstance(test, dict):
        return ("", "", "", "")
    t = get_str(test, "_type")
    if t == "Call" and _is_isinstance_call(test):
        tgt, tname = _isinstance_target_and_type(test)
        if tgt and tname:
            return (tgt, "isinstance", tname, "")
    if t == "Compare":
        none_tgt, none_kind = _is_none_compare(test)
        if none_tgt and none_kind:
            return (none_tgt, none_kind, "", "")
        cf_obj, cf_field, cf_value = _is_const_field_compare(test)
        if cf_obj and cf_field and cf_value:
            return (cf_obj, "const_field", cf_value, cf_field)
    return ("", "", "", "")


def _build_condition(
    test: ASTNode, prev_id: int, graph: FlowGraph, aliases: dict[str, int]
) -> tuple[int, int]:
    """Build narrow/widen pair for a condition.

    Returns (true_branch_id, false_branch_id).
    """
    target, narrow_type, type_name, field_name = _extract_condition_info(test)
    if not target and isinstance(test, dict):
        t = get_str(test, "_type")
        if t == "Name":
            name = get_str(test, "id")
            alias_id = aliases.get(name)
            if alias_id is not None:
                alias_node = graph.node_at(alias_id)
                if alias_node is not None and isinstance(alias_node, FlowCondAlias):
                    target = alias_node.target
                    narrow_type = alias_node.narrow_type
                    type_name = alias_node.type_name
                    field_name = alias_node.field_name
            if not target and name:
                target = name
                narrow_type = "truthy"
                type_name = ""
                field_name = ""
        if t == "UnaryOp":
            op = get_node(test, "op")
            if get_str(op, "_type") == "Not":
                operand = get_node(test, "operand")
                if operand:
                    true_id, false_id = _build_condition(
                        operand, prev_id, graph, aliases
                    )
                    return (false_id, true_id)
    if target and narrow_type:
        narrow_id = graph.next_id()
        narrow = FlowNarrow(
            id=narrow_id,
            prev=[prev_id],
            target=target,
            narrow_type=narrow_type,
            type_name=type_name,
            field_name=field_name,
        )
        graph.add(narrow)
        widen_id = graph.next_id()
        widen = FlowWiden(id=widen_id, prev=[prev_id], narrow_id=narrow_id)
        graph.add(widen)
        return (narrow_id, widen_id)
    nid = graph.next_id()
    join_true = FlowJoin(id=nid, prev=[prev_id])
    graph.add(join_true)
    nid2 = graph.next_id()
    join_false = FlowJoin(id=nid2, prev=[prev_id])
    graph.add(join_false)
    return (nid, nid2)


def _build_if(stmt: ASTNode, prev_id: int, graph: FlowGraph) -> int:
    """Build CFG for if/elif/else."""
    test = get_node(stmt, "test")
    body = get_nodes(stmt, "body")
    orelse = get_nodes(stmt, "orelse")
    aliases = _collect_aliases(graph)
    true_id = prev_id
    false_id = prev_id
    if test:
        true_id, false_id = _build_condition(test, prev_id, graph, aliases)
    then_end = _build_stmts(body, true_id, graph)
    else_end = false_id
    if orelse:
        else_end = _build_stmts(orelse, false_id, graph)
    then_node = graph.node_at(then_end)
    else_node = graph.node_at(else_end)
    then_dead = then_node is not None and isinstance(then_node, FlowUnreachable)
    else_dead = else_node is not None and isinstance(else_node, FlowUnreachable)
    if then_dead and else_dead:
        nid = graph.next_id()
        unr = FlowUnreachable(id=nid, prev=[then_end, else_end])
        graph.add(unr)
        return nid
    if then_dead:
        return else_end
    if else_dead:
        return then_end
    join_id = graph.next_id()
    join = FlowJoin(id=join_id, prev=[then_end, else_end])
    graph.add(join)
    return join_id


def _build_while(stmt: ASTNode, prev_id: int, graph: FlowGraph) -> int:
    """Build CFG for while loop."""
    test = get_node(stmt, "test")
    body = get_nodes(stmt, "body")
    head_id = graph.next_id()
    head = FlowLoopHead(id=head_id, prev=[prev_id])
    graph.add(head)
    aliases = _collect_aliases(graph)
    true_id = head_id
    false_id = head_id
    if test:
        true_id, false_id = _build_condition(test, head_id, graph, aliases)
    body_end = _build_stmts(body, true_id, graph)
    body_node = graph.node_at(body_end)
    if body_node is None or not isinstance(body_node, FlowUnreachable):
        head.prev.append(body_end)
    return false_id


def _build_for(stmt: ASTNode, prev_id: int, graph: FlowGraph) -> int:
    """Build CFG for for loop."""
    target = get_node(stmt, "target")
    body = get_nodes(stmt, "body")
    lineno = get_int(stmt, "lineno")
    head_id = graph.next_id()
    head = FlowLoopHead(id=head_id, prev=[prev_id])
    graph.add(head)
    assign_id = head_id
    if target and get_str(target, "_type") == "Name":
        name = get_str(target, "id")
        if name:
            assign_id = graph.next_id()
            assign = FlowAssign(id=assign_id, prev=[head_id], name=name, lineno=lineno)
            graph.add(assign)
    body_end = _build_stmts(body, assign_id, graph)
    body_node = graph.node_at(body_end)
    if body_node is None or not isinstance(body_node, FlowUnreachable):
        head.prev.append(body_end)
    join_id = graph.next_id()
    join = FlowJoin(id=join_id, prev=[head_id])
    graph.add(join)
    return join_id


def _build_assert(stmt: ASTNode, prev_id: int, graph: FlowGraph) -> int:
    """Build CFG for assert — narrows the subsequent code."""
    test = get_node(stmt, "test")
    if not test:
        return prev_id
    aliases = _collect_aliases(graph)
    true_id, _false_id = _build_condition(test, prev_id, graph, aliases)
    return true_id


def _build_unreachable(stmt: ASTNode, prev_id: int, graph: FlowGraph) -> int:
    """Build unreachable node after return/raise/break/continue."""
    nid = graph.next_id()
    unr = FlowUnreachable(id=nid, prev=[prev_id])
    graph.add(unr)
    return nid


def _collect_aliases(graph: FlowGraph) -> dict[str, int]:
    """Collect all condition aliases defined so far."""
    aliases: dict[str, int] = {}
    for node in graph.nodes:
        if isinstance(node, FlowCondAlias):
            aliases[node.alias_name] = node.id
    return aliases


# ---------------------------------------------------------------------------
# Forward-flow type resolver
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Narrowing / widening helpers
# ---------------------------------------------------------------------------


def _apply_narrow_type(
    cur_type: TypeNode,
    narrow_node: FlowNarrow,
    known_classes: dict[str, str],
) -> TypeNode:
    """Apply true-branch narrowing to a type."""
    from .types import OptionalType
    from .typecollect import py_type_to_type_dict, TypeCollectError

    if narrow_node.narrow_type == "isinstance":
        sig_errors: list[TypeCollectError] = []
        narrowed = py_type_to_type_dict(
            narrow_node.type_name, known_classes, sig_errors, 0, 0
        )
        return narrowed
    if narrow_node.narrow_type == "is_none":
        return cur_type
    if narrow_node.narrow_type == "is_not_none":
        if isinstance(cur_type, OptionalType):
            return cur_type.inner
        return cur_type
    if narrow_node.narrow_type == "truthy":
        if isinstance(cur_type, OptionalType):
            return cur_type.inner
        return cur_type
    return cur_type


def _apply_widen_type(
    cur_type: TypeNode,
    narrow_node: FlowNarrow,
    known_classes: dict[str, str],
) -> TypeNode:
    """Apply false-branch (inverse) narrowing to a type."""
    from .types import OptionalType, remove_from_union
    from .typecollect import py_type_to_type_dict, TypeCollectError

    if narrow_node.narrow_type == "isinstance":
        sig_errors: list[TypeCollectError] = []
        remove_t = py_type_to_type_dict(
            narrow_node.type_name, known_classes, sig_errors, 0, 0
        )
        return remove_from_union(cur_type, [remove_t])
    if narrow_node.narrow_type == "is_none":
        if isinstance(cur_type, OptionalType):
            return cur_type.inner
        return cur_type
    if narrow_node.narrow_type == "is_not_none":
        return cur_type
    if narrow_node.narrow_type == "truthy":
        return cur_type
    return cur_type


# ---------------------------------------------------------------------------
# Backward-walk type resolver
# ---------------------------------------------------------------------------


def _walk_prevs(
    graph: FlowGraph,
    prev_ids: list[int],
    variable: str,
    initial_types: dict[str, TypeNode],
    assigned_types: dict[int, TypeNode],
    known_classes: dict[str, str],
    cache: dict[str, TypeNode],
    visiting: dict[str, bool],
    depth: int,
) -> TypeNode | None:
    """Walk backward through predecessors and merge results."""
    if not prev_ids:
        return None
    if len(prev_ids) == 1:
        return _walk_back(
            graph,
            prev_ids[0],
            variable,
            initial_types,
            assigned_types,
            known_classes,
            cache,
            visiting,
            depth,
        )
    variants: list[TypeNode] = []
    for prev_id in prev_ids:
        t = _walk_back(
            graph,
            prev_id,
            variable,
            initial_types,
            assigned_types,
            known_classes,
            cache,
            visiting,
            depth,
        )
        if t is not None:
            dup = False
            for v in variants:
                if type_eq(v, t):
                    dup = True
            if not dup:
                variants.append(t)
    if not variants:
        return None
    if len(variants) == 1:
        return variants[0]
    return combine_types(variants)


def _walk_loop_head(
    graph: FlowGraph,
    node: FlowLoopHead,
    variable: str,
    initial_types: dict[str, TypeNode],
    assigned_types: dict[int, TypeNode],
    known_classes: dict[str, str],
    cache: dict[str, TypeNode],
    visiting: dict[str, bool],
    depth: int,
) -> TypeNode | None:
    """Fixed-point iteration for loop headers."""
    if not node.prev:
        return None
    entry_t = _walk_back(
        graph,
        node.prev[0],
        variable,
        initial_types,
        assigned_types,
        known_classes,
        cache,
        visiting,
        depth,
    )
    if len(node.prev) < 2:
        return entry_t
    cur = entry_t
    cache_key = variable + ":" + str(node.id)
    max_passes = 3
    p = 0
    while p < max_passes:
        if cur is not None:
            cache[cache_key] = cur
        variants: list[TypeNode] = []
        if cur is not None:
            variants.append(cur)
        bi = 1
        while bi < len(node.prev):
            bt = _walk_back(
                graph,
                node.prev[bi],
                variable,
                initial_types,
                assigned_types,
                known_classes,
                cache,
                visiting,
                depth,
            )
            if bt is not None:
                dup = False
                vi = 0
                while vi < len(variants):
                    if type_eq(variants[vi], bt):
                        dup = True
                    vi += 1
                if not dup:
                    variants.append(bt)
            bi += 1
        if not variants:
            new_t: TypeNode | None = None
        elif len(variants) == 1:
            new_t = variants[0]
        else:
            new_t = combine_types(variants)
        if cur is not None and new_t is not None and type_eq(cur, new_t):
            return new_t
        cur = new_t
        p += 1
    return cur


def _walk_back(
    graph: FlowGraph,
    node_id: int,
    variable: str,
    initial_types: dict[str, TypeNode],
    assigned_types: dict[int, TypeNode],
    known_classes: dict[str, str],
    cache: dict[str, TypeNode],
    visiting: dict[str, bool],
    depth: int,
) -> TypeNode | None:
    """Recursive backward walker — resolves the type of variable at node_id."""
    if depth > 200:
        return None
    cache_key = variable + ":" + str(node_id)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    if visiting.get(cache_key) is True:
        return cache.get(cache_key)
    visiting[cache_key] = True
    node = graph.node_at(node_id)
    if node is None:
        visiting[cache_key] = False
        return None
    result: TypeNode | None = None
    if isinstance(node, FlowStart):
        result = initial_types.get(variable)
    elif isinstance(node, FlowAssign):
        if node.name == variable:
            result = assigned_types.get(node.id)
        else:
            result = _walk_prevs(
                graph,
                node.prev,
                variable,
                initial_types,
                assigned_types,
                known_classes,
                cache,
                visiting,
                depth + 1,
            )
    elif isinstance(node, FlowCondAlias):
        result = _walk_prevs(
            graph,
            node.prev,
            variable,
            initial_types,
            assigned_types,
            known_classes,
            cache,
            visiting,
            depth + 1,
        )
    elif isinstance(node, FlowNarrow):
        prev_t = _walk_prevs(
            graph,
            node.prev,
            variable,
            initial_types,
            assigned_types,
            known_classes,
            cache,
            visiting,
            depth + 1,
        )
        if node.target == variable and prev_t is not None:
            result = _apply_narrow_type(prev_t, node, known_classes)
        else:
            result = prev_t
    elif isinstance(node, FlowWiden):
        narrow_node = graph.node_at(node.narrow_id)
        prev_t = _walk_prevs(
            graph,
            node.prev,
            variable,
            initial_types,
            assigned_types,
            known_classes,
            cache,
            visiting,
            depth + 1,
        )
        if (
            narrow_node is not None
            and isinstance(narrow_node, FlowNarrow)
            and narrow_node.target == variable
            and prev_t is not None
        ):
            result = _apply_widen_type(prev_t, narrow_node, known_classes)
        else:
            result = prev_t
    elif isinstance(node, FlowJoin):
        result = _walk_prevs(
            graph,
            node.prev,
            variable,
            initial_types,
            assigned_types,
            known_classes,
            cache,
            visiting,
            depth + 1,
        )
    elif isinstance(node, FlowLoopHead):
        result = _walk_loop_head(
            graph,
            node,
            variable,
            initial_types,
            assigned_types,
            known_classes,
            cache,
            visiting,
            depth + 1,
        )
    elif isinstance(node, FlowUnreachable):
        result = _walk_prevs(
            graph,
            node.prev,
            variable,
            initial_types,
            assigned_types,
            known_classes,
            cache,
            visiting,
            depth + 1,
        )
    visiting[cache_key] = False
    if result is not None:
        cache[cache_key] = result
    return result


def get_type_at(
    graph: FlowGraph,
    node_id: int,
    variable: str,
    initial_types: dict[str, TypeNode],
    assigned_types: dict[int, TypeNode],
    known_classes: dict[str, str],
) -> TypeNode | None:
    """Query the type of a variable at a specific flow node via backward walk."""
    cache: dict[str, TypeNode] = {}
    visiting: dict[str, bool] = {}
    return _walk_back(
        graph,
        node_id,
        variable,
        initial_types,
        assigned_types,
        known_classes,
        cache,
        visiting,
        0,
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def lookup_alias(graph: FlowGraph, name: str) -> FlowCondAlias | None:
    """Find the FlowCondAlias for a given alias variable name."""
    i = len(graph.nodes) - 1
    while i >= 0:
        node = graph.nodes[i]
        if isinstance(node, FlowCondAlias) and node.alias_name == name:
            return node
        i -= 1
    return None
