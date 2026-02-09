"""Serialization of IR objects to JSON-compatible dicts."""

from __future__ import annotations

from .ir import (
    AddrOf,
    Args,
    Array,
    Assert,
    Assign,
    BinaryOp,
    Block,
    BoolLit,
    Break,
    Bytes,
    Call,
    Cast,
    CatchClause,
    ChainedCompare,
    Char,
    CharAt,
    CharClassify,
    CharLen,
    CharLit,
    CharSequence,
    CompGenerator,
    Constant,
    Continue,
    DictComp,
    EntryPoint,
    Enum,
    EnumVariant,
    Export,
    Expr,
    ExprStmt,
    Field,
    FieldAccess,
    FieldInfo,
    FieldLV,
    FloatLit,
    ForClassic,
    ForRange,
    FuncInfo,
    FuncRef,
    FuncType,
    Function,
    GetEnv,
    If,
    Index,
    IndexLV,
    IntLit,
    IntToStr,
    InterfaceDef,
    InterfaceRef,
    IsNil,
    IsType,
    LValue,
    LastElement,
    Len,
    ListComp,
    Loc,
    MakeMap,
    MakeSlice,
    Map,
    MapLit,
    Match,
    MatchCase,
    MaxExpr,
    MethodCall,
    MethodSig,
    MinExpr,
    Module,
    NilLit,
    NoOp,
    OpAssign,
    Optional,
    OwnershipInfo,
    Param,
    ParamInfo,
    ParseInt,
    Pointer,
    Primitive,
    Print,
    Raise,
    ReadAll,
    ReadBytes,
    ReadBytesN,
    ReadLine,
    Receiver,
    Return,
    SentinelToOptional,
    Set,
    SetComp,
    SetLit,
    Slice,
    SliceConvert,
    SliceExpr,
    SliceLV,
    SliceLit,
    SoftFail,
    StaticCall,
    Stmt,
    StringConcat,
    StringFormat,
    StringLit,
    Struct,
    StructInfo,
    StructLit,
    StructRef,
    Substring,
    SymbolTable,
    Ternary,
    TrimChars,
    TryCatch,
    Truthy,
    Tuple,
    TupleAssign,
    TupleLit,
    Type,
    TypeAssert,
    TypeCase,
    TypeSwitch,
    DerefLV,
    Union,
    UnaryOp,
    Var,
    VarDecl,
    VarLV,
    WeakRef,
    While,
    WriteBytes,
)


def serialize(obj: object) -> object:
    """Recursively serialize an object to a JSON-compatible structure."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple)):
        return [serialize(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        items: list[object] = [serialize(x) for x in obj]
        try:
            items.sort()
        except TypeError:
            pass
        return items
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    return _ir_serialize(obj)


def _ir_serialize(obj: object) -> object:
    """Serialize IR types via isinstance dispatch."""
    if isinstance(obj, Type):
        return _serialize_type(obj)
    if isinstance(obj, Expr):
        return _serialize_expr(obj)
    if isinstance(obj, Stmt):
        return _serialize_stmt(obj)
    if isinstance(obj, LValue):
        return _serialize_lvalue(obj)
    if isinstance(obj, Loc):
        return {
            "_type": "Loc",
            "line": obj.line,
            "col": obj.col,
            "end_line": obj.end_line,
            "end_col": obj.end_col,
        }
    if isinstance(obj, CompGenerator):
        return {
            "_type": "CompGenerator",
            "targets": serialize(obj.targets),
            "iterable": serialize(obj.iterable),
            "conditions": serialize(obj.conditions),
        }
    if isinstance(obj, TypeCase):
        return {
            "_type": "TypeCase",
            "typ": serialize(obj.typ),
            "body": serialize(obj.body),
            "loc": serialize(obj.loc),
        }
    if isinstance(obj, MatchCase):
        return {
            "_type": "MatchCase",
            "patterns": serialize(obj.patterns),
            "body": serialize(obj.body),
            "loc": serialize(obj.loc),
        }
    if isinstance(obj, CatchClause):
        return {
            "_type": "CatchClause",
            "var": serialize(obj.var),
            "typ": serialize(obj.typ),
            "body": serialize(obj.body),
        }
    if isinstance(obj, Receiver):
        return {
            "_type": "Receiver",
            "name": obj.name,
            "typ": serialize(obj.typ),
            "mutable": obj.mutable,
            "pointer": obj.pointer,
        }
    if isinstance(obj, Param):
        return {
            "_type": "Param",
            "name": obj.name,
            "typ": serialize(obj.typ),
            "default": serialize(obj.default),
            "mutable": obj.mutable,
            "loc": serialize(obj.loc),
            "is_modified": obj.is_modified,
            "is_unused": obj.is_unused,
            "ownership": obj.ownership,
        }
    if isinstance(obj, Field):
        return {
            "_type": "Field",
            "name": obj.name,
            "typ": serialize(obj.typ),
            "default": serialize(obj.default),
            "loc": serialize(obj.loc),
            "ownership": obj.ownership,
        }
    if isinstance(obj, Function):
        return {
            "_type": "Function",
            "name": obj.name,
            "params": serialize(obj.params),
            "ret": serialize(obj.ret),
            "body": serialize(obj.body),
            "doc": obj.doc,
            "receiver": serialize(obj.receiver),
            "fallible": obj.fallible,
            "loc": serialize(obj.loc),
            "needs_named_returns": obj.needs_named_returns,
            "rune_vars": serialize(obj.rune_vars),
        }
    if isinstance(obj, Struct):
        return {
            "_type": "Struct",
            "name": obj.name,
            "doc": obj.doc,
            "fields": serialize(obj.fields),
            "methods": serialize(obj.methods),
            "implements": serialize(obj.implements),
            "loc": serialize(obj.loc),
            "is_exception": obj.is_exception,
            "embedded_type": obj.embedded_type,
            "const_fields": serialize(obj.const_fields),
        }
    if isinstance(obj, Module):
        return {
            "_type": "Module",
            "name": obj.name,
            "doc": obj.doc,
            "structs": serialize(obj.structs),
            "interfaces": serialize(obj.interfaces),
            "functions": serialize(obj.functions),
            "constants": serialize(obj.constants),
            "enums": serialize(obj.enums),
            "exports": serialize(obj.exports),
            "statements": serialize(obj.statements),
            "hierarchy_root": obj.hierarchy_root,
            "entrypoint": serialize(obj.entrypoint),
        }
    if isinstance(obj, InterfaceDef):
        return {
            "_type": "InterfaceDef",
            "name": obj.name,
            "methods": serialize(obj.methods),
            "fields": serialize(obj.fields),
            "loc": serialize(obj.loc),
        }
    if isinstance(obj, MethodSig):
        return {
            "_type": "MethodSig",
            "name": obj.name,
            "params": serialize(obj.params),
            "ret": serialize(obj.ret),
            "loc": serialize(obj.loc),
        }
    if isinstance(obj, Enum):
        return {
            "_type": "Enum",
            "name": obj.name,
            "variants": serialize(obj.variants),
            "loc": serialize(obj.loc),
        }
    if isinstance(obj, EnumVariant):
        return {
            "_type": "EnumVariant",
            "name": obj.name,
            "value": serialize(obj.value),
            "loc": serialize(obj.loc),
        }
    if isinstance(obj, Export):
        return {
            "_type": "Export",
            "name": obj.name,
            "kind": obj.kind,
            "loc": serialize(obj.loc),
        }
    if isinstance(obj, Constant):
        return {
            "_type": "Constant",
            "name": obj.name,
            "typ": serialize(obj.typ),
            "value": serialize(obj.value),
            "loc": serialize(obj.loc),
        }
    if isinstance(obj, EntryPoint):
        return {
            "_type": "EntryPoint",
            "function_name": obj.function_name,
            "loc": serialize(obj.loc),
        }
    if isinstance(obj, SymbolTable):
        return {
            "_type": "SymbolTable",
            "structs": serialize(obj.structs),
            "functions": serialize(obj.functions),
            "constants": serialize(obj.constants),
            "field_to_structs": serialize(obj.field_to_structs),
            "method_to_structs": serialize(obj.method_to_structs),
        }
    if isinstance(obj, StructInfo):
        return {
            "_type": "StructInfo",
            "name": obj.name,
            "fields": serialize(obj.fields),
            "methods": serialize(obj.methods),
            "is_node": obj.is_node,
            "is_exception": obj.is_exception,
            "bases": serialize(obj.bases),
            "init_params": serialize(obj.init_params),
            "param_to_field": serialize(obj.param_to_field),
            "needs_constructor": obj.needs_constructor,
            "const_fields": serialize(obj.const_fields),
        }
    if isinstance(obj, FieldInfo):
        d = {
            "_type": "FieldInfo",
            "name": obj.name,
            "typ": serialize(obj.typ),
            "py_name": obj.py_name,
            "has_default": obj.has_default,
        }
        if obj.default is not None:
            d["default"] = serialize(obj.default)
        return d
    if isinstance(obj, FuncInfo):
        return {
            "_type": "FuncInfo",
            "name": obj.name,
            "params": serialize(obj.params),
            "return_type": serialize(obj.return_type),
            "is_method": obj.is_method,
            "receiver_type": obj.receiver_type,
        }
    if isinstance(obj, ParamInfo):
        return {
            "_type": "ParamInfo",
            "name": obj.name,
            "typ": serialize(obj.typ),
            "has_default": obj.has_default,
            "default_value": serialize(obj.default_value),
            "modifier": obj.modifier,
        }
    if isinstance(obj, OwnershipInfo):
        return {
            "_type": "OwnershipInfo",
            "escaping_vars": serialize(obj.escaping_vars),
            "shared_vars": serialize(obj.shared_vars),
            "weak_fields": serialize(obj.weak_fields),
        }
    return "<unserializable>"


def _serialize_type(obj: Type) -> dict[str, object]:
    """Serialize Type subclasses."""
    if isinstance(obj, Primitive):
        return {"_type": "Primitive", "kind": obj.kind}
    if isinstance(obj, Char):
        return {"_type": "Char"}
    if isinstance(obj, CharSequence):
        return {"_type": "CharSequence"}
    if isinstance(obj, Bytes):
        return {"_type": "Bytes", "kind": "bytes"}
    if isinstance(obj, Slice):
        d: dict[str, object] = {"_type": "Slice", "element": serialize(obj.element)}
        if isinstance(obj.element, Primitive) and obj.element.kind == "byte":
            d["kind"] = "bytes"
        return d
    if isinstance(obj, Array):
        return {"_type": "Array", "element": serialize(obj.element), "size": obj.size}
    if isinstance(obj, Map):
        return {
            "_type": "Map",
            "key": serialize(obj.key),
            "value": serialize(obj.value),
        }
    if isinstance(obj, Set):
        return {"_type": "Set", "element": serialize(obj.element)}
    if isinstance(obj, Tuple):
        result: dict[str, object] = {
            "_type": "Tuple",
            "elements": serialize(obj.elements),
        }
        if obj.variadic:
            result["variadic"] = True
        return result
    if isinstance(obj, Pointer):
        return {"_type": "Pointer", "target": serialize(obj.target), "owned": obj.owned}
    if isinstance(obj, Optional):
        return {"_type": "Optional", "inner": serialize(obj.inner)}
    if isinstance(obj, StructRef):
        return {"_type": "StructRef", "name": obj.name}
    if isinstance(obj, InterfaceRef):
        return {"_type": "InterfaceRef", "name": obj.name}
    if isinstance(obj, Union):
        return {"_type": "Union", "name": obj.name, "variants": serialize(obj.variants)}
    if isinstance(obj, FuncType):
        return {
            "_type": "FuncType",
            "params": serialize(obj.params),
            "ret": serialize(obj.ret),
            "captures": obj.captures,
            "receiver": serialize(obj.receiver),
        }
    return {"_type": "Type"}


def _serialize_expr(obj: Expr) -> dict[str, object]:
    """Serialize Expr subclasses."""
    d: dict[str, object] = {
        "typ": serialize(obj.typ),
        "loc": serialize(obj.loc),
        "is_interface": obj.is_interface,
        "narrowed_type": serialize(obj.narrowed_type),
        "escapes": obj.escapes,
    }
    if isinstance(obj, IntLit):
        d["_type"] = "IntLit"
        d["value"] = obj.value
        d["format"] = obj.format
    elif isinstance(obj, FloatLit):
        d["_type"] = "FloatLit"
        d["value"] = obj.value
        d["format"] = obj.format
    elif isinstance(obj, StringLit):
        d["_type"] = "StringLit"
        d["value"] = obj.value
    elif isinstance(obj, CharLit):
        d["_type"] = "CharLit"
        d["value"] = obj.value
    elif isinstance(obj, BoolLit):
        d["_type"] = "BoolLit"
        d["value"] = obj.value
    elif isinstance(obj, NilLit):
        d["_type"] = "NilLit"
    elif isinstance(obj, Var):
        d["_type"] = "Var"
        d["name"] = obj.name
    elif isinstance(obj, FieldAccess):
        d["_type"] = "FieldAccess"
        d["obj"] = serialize(obj.obj)
        d["field"] = obj.field
        d["through_pointer"] = obj.through_pointer
    elif isinstance(obj, Index):
        d["_type"] = "Index"
        d["obj"] = serialize(obj.obj)
        d["index"] = serialize(obj.index)
        d["bounds_check"] = obj.bounds_check
        d["returns_optional"] = obj.returns_optional
    elif isinstance(obj, SliceExpr):
        d["_type"] = "SliceExpr"
        d["obj"] = serialize(obj.obj)
        d["low"] = serialize(obj.low)
        d["high"] = serialize(obj.high)
        d["step"] = serialize(obj.step)
    elif isinstance(obj, Call):
        d["_type"] = "Call"
        d["func"] = obj.func
        d["args"] = serialize(obj.args)
        d["reverse"] = obj.reverse
    elif isinstance(obj, MethodCall):
        d["_type"] = "MethodCall"
        d["obj"] = serialize(obj.obj)
        d["method"] = obj.method
        d["args"] = serialize(obj.args)
        d["receiver_type"] = serialize(obj.receiver_type)
        d["reverse"] = obj.reverse
    elif isinstance(obj, StaticCall):
        d["_type"] = "StaticCall"
        d["on_type"] = serialize(obj.on_type)
        d["method"] = obj.method
        d["args"] = serialize(obj.args)
    elif isinstance(obj, FuncRef):
        d["_type"] = "FuncRef"
        d["name"] = obj.name
        d["obj"] = serialize(obj.obj)
    elif isinstance(obj, BinaryOp):
        d["_type"] = "BinaryOp"
        d["op"] = obj.op
        d["left"] = serialize(obj.left)
        d["right"] = serialize(obj.right)
    elif isinstance(obj, UnaryOp):
        d["_type"] = "UnaryOp"
        d["op"] = obj.op
        d["operand"] = serialize(obj.operand)
    elif isinstance(obj, Truthy):
        d["_type"] = "Truthy"
        d["expr"] = serialize(obj.expr)
    elif isinstance(obj, Ternary):
        d["_type"] = "Ternary"
        d["cond"] = serialize(obj.cond)
        d["then_expr"] = serialize(obj.then_expr)
        d["else_expr"] = serialize(obj.else_expr)
        d["needs_statement"] = obj.needs_statement
    elif isinstance(obj, ChainedCompare):
        d["_type"] = "ChainedCompare"
        d["operands"] = serialize(obj.operands)
        d["ops"] = serialize(obj.ops)
    elif isinstance(obj, MinExpr):
        d["_type"] = "MinExpr"
        d["left"] = serialize(obj.left)
        d["right"] = serialize(obj.right)
    elif isinstance(obj, MaxExpr):
        d["_type"] = "MaxExpr"
        d["left"] = serialize(obj.left)
        d["right"] = serialize(obj.right)
    elif isinstance(obj, Cast):
        d["_type"] = "Cast"
        d["expr"] = serialize(obj.expr)
        d["to_type"] = serialize(obj.to_type)
    elif isinstance(obj, TypeAssert):
        d["_type"] = "TypeAssert"
        d["expr"] = serialize(obj.expr)
        d["asserted"] = serialize(obj.asserted)
        d["safe"] = obj.safe
    elif isinstance(obj, IsType):
        d["_type"] = "IsType"
        d["expr"] = serialize(obj.expr)
        d["tested_type"] = serialize(obj.tested_type)
    elif isinstance(obj, IsNil):
        d["_type"] = "IsNil"
        d["expr"] = serialize(obj.expr)
        d["negated"] = obj.negated
    elif isinstance(obj, Len):
        d["_type"] = "Len"
        d["expr"] = serialize(obj.expr)
    elif isinstance(obj, MakeSlice):
        d["_type"] = "MakeSlice"
        d["element_type"] = serialize(obj.element_type)
        d["length"] = serialize(obj.length)
        d["capacity"] = serialize(obj.capacity)
    elif isinstance(obj, MakeMap):
        d["_type"] = "MakeMap"
        d["key_type"] = serialize(obj.key_type)
        d["value_type"] = serialize(obj.value_type)
    elif isinstance(obj, SliceLit):
        d["_type"] = "SliceLit"
        d["element_type"] = serialize(obj.element_type)
        d["elements"] = serialize(obj.elements)
    elif isinstance(obj, MapLit):
        d["_type"] = "MapLit"
        d["key_type"] = serialize(obj.key_type)
        d["value_type"] = serialize(obj.value_type)
        d["entries"] = serialize(obj.entries)
    elif isinstance(obj, SetLit):
        d["_type"] = "SetLit"
        d["element_type"] = serialize(obj.element_type)
        d["elements"] = serialize(obj.elements)
    elif isinstance(obj, TupleLit):
        d["_type"] = "TupleLit"
        d["elements"] = serialize(obj.elements)
    elif isinstance(obj, StructLit):
        d["_type"] = "StructLit"
        d["struct_name"] = obj.struct_name
        d["fields"] = serialize(obj.fields)
        d["embedded_value"] = serialize(obj.embedded_value)
    elif isinstance(obj, LastElement):
        d["_type"] = "LastElement"
        d["sequence"] = serialize(obj.sequence)
    elif isinstance(obj, SliceConvert):
        d["_type"] = "SliceConvert"
        d["source"] = serialize(obj.source)
        d["target_element_type"] = serialize(obj.target_element_type)
    elif isinstance(obj, CharAt):
        d["_type"] = "CharAt"
        d["string"] = serialize(obj.string)
        d["index"] = serialize(obj.index)
    elif isinstance(obj, CharLen):
        d["_type"] = "CharLen"
        d["string"] = serialize(obj.string)
    elif isinstance(obj, Substring):
        d["_type"] = "Substring"
        d["string"] = serialize(obj.string)
        d["low"] = serialize(obj.low)
        d["high"] = serialize(obj.high)
    elif isinstance(obj, StringConcat):
        d["_type"] = "StringConcat"
        d["parts"] = serialize(obj.parts)
    elif isinstance(obj, StringFormat):
        d["_type"] = "StringFormat"
        d["template"] = obj.template
        d["args"] = serialize(obj.args)
    elif isinstance(obj, TrimChars):
        d["_type"] = "TrimChars"
        d["string"] = serialize(obj.string)
        d["chars"] = serialize(obj.chars)
        d["mode"] = obj.mode
    elif isinstance(obj, CharClassify):
        d["_type"] = "CharClassify"
        d["kind"] = obj.kind
        d["char"] = serialize(obj.char)
    elif isinstance(obj, ParseInt):
        d["_type"] = "ParseInt"
        d["string"] = serialize(obj.string)
        d["base"] = serialize(obj.base)
    elif isinstance(obj, IntToStr):
        d["_type"] = "IntToStr"
        d["value"] = serialize(obj.value)
    elif isinstance(obj, SentinelToOptional):
        d["_type"] = "SentinelToOptional"
        d["expr"] = serialize(obj.expr)
        d["sentinel"] = serialize(obj.sentinel)
    elif isinstance(obj, AddrOf):
        d["_type"] = "AddrOf"
        d["operand"] = serialize(obj.operand)
    elif isinstance(obj, WeakRef):
        d["_type"] = "WeakRef"
        d["operand"] = serialize(obj.operand)
    elif isinstance(obj, ReadLine):
        d["_type"] = "ReadLine"
    elif isinstance(obj, ReadAll):
        d["_type"] = "ReadAll"
    elif isinstance(obj, ReadBytes):
        d["_type"] = "ReadBytes"
    elif isinstance(obj, ReadBytesN):
        d["_type"] = "ReadBytesN"
        d["count"] = serialize(obj.count)
    elif isinstance(obj, WriteBytes):
        d["_type"] = "WriteBytes"
        d["data"] = serialize(obj.data)
        d["stderr"] = obj.stderr
    elif isinstance(obj, Args):
        d["_type"] = "Args"
    elif isinstance(obj, GetEnv):
        d["_type"] = "GetEnv"
        d["name"] = serialize(obj.name)
        d["default"] = serialize(obj.default)
    elif isinstance(obj, ListComp):
        d["_type"] = "ListComp"
        d["element"] = serialize(obj.element)
        d["generators"] = serialize(obj.generators)
    elif isinstance(obj, SetComp):
        d["_type"] = "SetComp"
        d["element"] = serialize(obj.element)
        d["generators"] = serialize(obj.generators)
    elif isinstance(obj, DictComp):
        d["_type"] = "DictComp"
        d["key"] = serialize(obj.key)
        d["value"] = serialize(obj.value)
        d["generators"] = serialize(obj.generators)
    else:
        d["_type"] = "Expr"
    return d


def _serialize_stmt(obj: Stmt) -> dict[str, object]:
    """Serialize Stmt subclasses."""
    d: dict[str, object] = {"loc": serialize(obj.loc)}
    if isinstance(obj, NoOp):
        d["_type"] = "NoOp"
    elif isinstance(obj, VarDecl):
        d["_type"] = "VarDecl"
        d["name"] = obj.name
        d["typ"] = serialize(obj.typ)
        d["value"] = serialize(obj.value)
        d["mutable"] = obj.mutable
        d["is_reassigned"] = obj.is_reassigned
        d["is_const"] = obj.is_const
        d["initial_value_unused"] = obj.initial_value_unused
        d["ownership"] = obj.ownership
        d["region"] = obj.region
    elif isinstance(obj, Assign):
        d["_type"] = "Assign"
        d["target"] = serialize(obj.target)
        d["value"] = serialize(obj.value)
        d["is_declaration"] = obj.is_declaration
        d["decl_typ"] = serialize(obj.decl_typ)
    elif isinstance(obj, TupleAssign):
        d["_type"] = "TupleAssign"
        d["targets"] = serialize(obj.targets)
        d["value"] = serialize(obj.value)
        d["is_declaration"] = obj.is_declaration
        d["unused_indices"] = serialize(obj.unused_indices)
        d["new_targets"] = serialize(obj.new_targets)
    elif isinstance(obj, OpAssign):
        d["_type"] = "OpAssign"
        d["target"] = serialize(obj.target)
        d["op"] = obj.op
        d["value"] = serialize(obj.value)
    elif isinstance(obj, ExprStmt):
        d["_type"] = "ExprStmt"
        d["expr"] = serialize(obj.expr)
    elif isinstance(obj, Return):
        d["_type"] = "Return"
        d["value"] = serialize(obj.value)
    elif isinstance(obj, If):
        d["_type"] = "If"
        d["cond"] = serialize(obj.cond)
        d["then_body"] = serialize(obj.then_body)
        d["else_body"] = serialize(obj.else_body)
        d["init"] = serialize(obj.init)
        d["hoisted_vars"] = serialize(obj.hoisted_vars)
    elif isinstance(obj, TypeSwitch):
        d["_type"] = "TypeSwitch"
        d["expr"] = serialize(obj.expr)
        d["binding"] = obj.binding
        d["cases"] = serialize(obj.cases)
        d["default"] = serialize(obj.default)
        d["binding_unused"] = obj.binding_unused
        d["binding_reassigned"] = obj.binding_reassigned
        d["hoisted_vars"] = serialize(obj.hoisted_vars)
    elif isinstance(obj, Match):
        d["_type"] = "Match"
        d["expr"] = serialize(obj.expr)
        d["cases"] = serialize(obj.cases)
        d["default"] = serialize(obj.default)
        d["hoisted_vars"] = serialize(obj.hoisted_vars)
    elif isinstance(obj, ForRange):
        d["_type"] = "ForRange"
        d["index"] = obj.index
        d["value"] = obj.value
        d["iterable"] = serialize(obj.iterable)
        d["body"] = serialize(obj.body)
        d["hoisted_vars"] = serialize(obj.hoisted_vars)
    elif isinstance(obj, ForClassic):
        d["_type"] = "ForClassic"
        d["init"] = serialize(obj.init)
        d["cond"] = serialize(obj.cond)
        d["post"] = serialize(obj.post)
        d["body"] = serialize(obj.body)
        d["hoisted_vars"] = serialize(obj.hoisted_vars)
    elif isinstance(obj, While):
        d["_type"] = "While"
        d["cond"] = serialize(obj.cond)
        d["body"] = serialize(obj.body)
        d["hoisted_vars"] = serialize(obj.hoisted_vars)
    elif isinstance(obj, Break):
        d["_type"] = "Break"
        d["label"] = obj.label
    elif isinstance(obj, Continue):
        d["_type"] = "Continue"
        d["label"] = obj.label
    elif isinstance(obj, Block):
        d["_type"] = "Block"
        d["body"] = serialize(obj.body)
        d["no_scope"] = obj.no_scope
    elif isinstance(obj, TryCatch):
        d["_type"] = "TryCatch"
        d["body"] = serialize(obj.body)
        d["catches"] = serialize(obj.catches)
        d["reraise"] = obj.reraise
        d["has_returns"] = obj.has_returns
        d["has_catch_returns"] = obj.has_catch_returns
        d["hoisted_vars"] = serialize(obj.hoisted_vars)
    elif isinstance(obj, Raise):
        d["_type"] = "Raise"
        d["error_type"] = obj.error_type
        d["message"] = serialize(obj.message)
        d["pos"] = serialize(obj.pos)
        d["reraise_var"] = obj.reraise_var
    elif isinstance(obj, Assert):
        d["_type"] = "Assert"
        d["test"] = serialize(obj.test)
        d["message"] = serialize(obj.message)
    elif isinstance(obj, SoftFail):
        d["_type"] = "SoftFail"
    elif isinstance(obj, Print):
        d["_type"] = "Print"
        d["value"] = serialize(obj.value)
        d["newline"] = obj.newline
        d["stderr"] = obj.stderr
    elif isinstance(obj, EntryPoint):
        d["_type"] = "EntryPoint"
        d["function_name"] = obj.function_name
    else:
        d["_type"] = "Stmt"
    return d


def _serialize_lvalue(obj: LValue) -> dict[str, object]:
    """Serialize LValue subclasses."""
    d: dict[str, object] = {"loc": serialize(obj.loc)}
    if isinstance(obj, VarLV):
        d["_type"] = "VarLV"
        d["name"] = obj.name
    elif isinstance(obj, FieldLV):
        d["_type"] = "FieldLV"
        d["obj"] = serialize(obj.obj)
        d["field"] = obj.field
    elif isinstance(obj, IndexLV):
        d["_type"] = "IndexLV"
        d["obj"] = serialize(obj.obj)
        d["index"] = serialize(obj.index)
    elif isinstance(obj, SliceLV):
        d["_type"] = "SliceLV"
        d["obj"] = serialize(obj.obj)
        d["low"] = serialize(obj.low)
        d["high"] = serialize(obj.high)
        d["step"] = serialize(obj.step)
    elif isinstance(obj, DerefLV):
        d["_type"] = "DerefLV"
        d["ptr"] = serialize(obj.ptr)
    else:
        d["_type"] = "LValue"
    return d


def signatures_to_dict(symbols: SymbolTable) -> dict[str, object]:
    """Serialize signatures phase output: {"functions": {...}, "methods": {...}}."""
    functions: dict[str, object] = {}
    for name, info in symbols.functions.items():
        functions[name] = serialize(info)
    methods: dict[str, object] = {}
    for sname, struct in symbols.structs.items():
        for mname, minfo in struct.methods.items():
            methods[sname + "." + mname] = serialize(minfo)
    return {"functions": functions, "methods": methods}


def fields_to_dict(symbols: SymbolTable) -> dict[str, object]:
    """Serialize fields phase output: {"classes": {...}}."""
    classes: dict[str, object] = {}
    for name, struct in symbols.structs.items():
        fields: dict[str, object] = {
            fname: serialize(finfo) for fname, finfo in struct.fields.items()
        }
        classes[name] = {
            "fields": fields,
            "init_params": struct.init_params,
            "is_dataclass": struct.is_dataclass,
            "kw_only": struct.kw_only,
        }
    return {"classes": classes}


def hierarchy_to_dict(
    symbols: SymbolTable, hierarchy_root: str | None
) -> dict[str, object]:
    """Serialize hierarchy phase output: {"ancestors": {...}, "root": ...}."""
    ancestors: dict[str, object] = {}
    for name, struct in symbols.structs.items():
        ancestors[name] = struct.bases
    return {"ancestors": ancestors, "root": hierarchy_root}


def module_to_dict(module: Module) -> object:
    """Serialize IR Module to dict."""
    return serialize(module)
