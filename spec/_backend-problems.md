# Backend Problems

This document tracks places where backends compensate for missing frontend/middleend functionality. Each item represents work that ideally would be done earlier in the pipeline.

---

## C

### 1. Aggregate Type Collection

**Location:** `c.py:1100-1513` (`_collect_tuple_types`, `_collect_slice_types`, `_visit_*_for_*`)

The backend manually traverses the entire IR module to collect all tuple and slice types, building dictionaries for typedef emission. This requires ~400 lines of visitor code duplicating AST traversal logic.

**Should be:** A middleend pass that computes `Module.used_tuple_types` and `Module.used_slice_types` sets.

---

### 2. Integer Literal in `any`-Typed Context

**Location:** `c.py:2101-2113`

```python
# Workaround: if type is 'any' but value is integer literal, use int64_t
if isinstance(typ, InterfaceRef) and typ.name == "any":
    if is_int_lit:
        typ = INT
```

When assigning an integer literal to a variable typed as `any`, the backend overrides the type to `int64_t` to avoid boxing overhead.

**Should be:** Inference should specialize `any`-typed assignments with concrete literal types, or lowering should emit explicit box/unbox operations.

---

### 3. Hoisted Variable Type Compatibility

**Location:** `c.py:2155-2165`, `2197-2207`

```python
# Workaround: if already declared with integer type and new value is compatible, don't redeclare
if (already_declared and not is_hoisted
    and self._hoisted_vars.get(var_name) in ("int64_t", "int32_t", "int", "bool")
    and (c_type in ("int64_t", "int32_t", "int", "bool", "Any *", "void *") ...)):
    needs_decl = False
```

The backend tracks hoisted variable types and avoids redeclaration when types are compatible (e.g., `int64_t` and `int32_t`). This logic is duplicated across `_emit_stmt_Assign` and `_emit_stmt_TupleAssign`.

**Should be:** The middleend hoisting pass should compute a single canonical type for each hoisted variable, resolving compatibility at analysis time.

---

### 4. `divmod` Special Case

**Location:** `c.py:2215-2241`

The backend detects `divmod()` calls during tuple unpacking and emits inline division/modulo operations. This pattern matching happens at emission time.

**Should be:** Lowering should transform `divmod(a, b)` into a `TupleLit` with `BinaryOp("/")` and `BinaryOp("%")` elements, or a dedicated `DivMod` IR node.

---

### 5. Interface Variable Tracking

**Location:** `c.py:1897-1904`, `1931-1938`, `2036-2038`, `2116-2117`

The backend maintains `_interface_vars: set[str]` to track which variables have interface types, enabling correct cast emission when assigning structs to interfaces.

**Should be:** IR `Var` and `VarDecl` nodes should have an `is_interface_typed` annotation from inference, or lowering should emit explicit `InterfaceCast` nodes.

---

### 6. Callback Parameter Receiver Tracking

**Location:** `c.py:1903-1904`, `1937-1938`, `3163-3164`

```python
if isinstance(p.typ, FuncType) and p.typ.receiver is not None:
    self._callback_params.add(p.name)
```

The backend tracks parameters with bound method types to prepend `self` when invoking them. This requires inspecting `FuncType.receiver` at emission time.

**Should be:** Lowering should emit `Call` nodes with the receiver pre-applied to the argument list for bound method invocations.

---

### 7. Escape Analysis String Copying

**Location:** `c.py:2094-2097`

```python
if isinstance(stmt.target, FieldLV) and stmt.value.escapes:
    if isinstance(value_typ, Primitive) and value_typ.kind == "string":
        value = f"arena_strdup(g_arena, {value})"
```

The backend checks `escapes` annotation and wraps string values in `arena_strdup` when storing to fields.

**Should be:** Lowering should emit an explicit `StringCopy` or `ArenaAlloc` node when escape analysis determines a copy is needed.

---

### 8. Kind String Fallback

**Location:** `c.py:1693-1706`

```python
def _struct_name_to_kind(self, name: str) -> str:
    if name in self._kind_cache:
        return self._kind_cache[name]
    # Fallback: convert PascalCase to kebab-case
    ...
```

The backend computes kind strings from struct names when `const_fields["kind"]` is missing. This duplicates naming convention logic.

**Should be:** Fields phase should always populate `const_fields["kind"]` for structs in a hierarchy, using the naming convention if not explicitly declared.

---

### 9. Rvalue Temps for Optional Slice Fields

**Location:** `c.py:3681-3740` (`_emit_rvalue_temps`)

The backend traverses expression trees looking for `StructLit` nodes that pass rvalue slices to `Optional(Slice)` fields, emitting temporary variables since C can't take addresses of rvalues.

**Should be:** Lowering should detect this pattern and emit explicit `VarDecl` + `AddrOf` when constructing structs with optional slice fields from rvalues.

---

### 10. Bool-to-Int Casting

**Location:** `c.py:3121-3137`, `3878-3896`

```python
if expr.args[0].typ == BOOL:
    base = f"(int64_t){base}"
```

The backend inserts `(int64_t)` casts when bool values are used in arithmetic contexts (`pow`, `divmod`, `min`, `max`).

**Should be:** Inference or lowering should insert explicit `Cast(BOOL, INT)` nodes when bools flow into int-expecting operations.

---

### 11. String Comparison Detection

**Location:** `c.py:3358-3374`

```python
left_is_str = isinstance(expr.left.typ, Primitive) and expr.left.typ.kind == "string"
if op == "==" and (left_is_str or right_is_str ...):
    return f"(strcmp({left}, {right}) == 0)"
```

The backend checks operand types to determine whether to use `strcmp` vs `==`.

**Should be:** Lowering should emit `StringCompare` IR nodes for string equality/inequality, distinct from primitive `BinaryOp("==")`.

---

### 12. Rune vs String Char Comparison

**Location:** `c.py:3326-3355`

The backend detects comparisons between runes and single-character string literals, emitting character literal comparisons instead of string comparisons.

**Should be:** Inference should normalize `rune == "x"` to `rune == ord("x")`, or lowering should emit `RuneCompare` nodes.

---

### 13. Truthy Type Dispatch

**Location:** `c.py:3821-3835`

```python
if inner_type == STRING:
    return f"({inner} != NULL && {inner}[0] != '\\0')"
if isinstance(inner_type, Slice):
    return f"({inner}.len > 0)"
```

The backend switches on the inner type of `Truthy` nodes to emit type-appropriate truthiness checks.

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `StringNonEmpty`, `SliceNonEmpty`, `IsNotNil`, etc.

---

### 14. Collection Method Polymorphism

**Location:** `c.py:3244-3274`

The backend checks whether `obj_type` is `Slice`, `Map`, or `Set` to emit different method implementations (e.g., `extend` as inline loop vs helper function).

**Should be:** Lowering should dispatch collection methods to type-specific IR nodes: `SliceExtend`, `MapGet`, etc.

---

### 15. Struct-to-Interface Casting

**Location:** `c.py:2043-2064`, `2121-2150`, `3169-3191`

Multiple locations check whether a struct value is being assigned/passed to an interface-typed target and insert C casts.

**Should be:** Lowering should emit explicit `InterfaceCast(expr, target_interface)` nodes when widening struct to interface.

---

### 16. Function Signature Collection

**Location:** `c.py:1089-1098` (`_collect_function_sigs`)

The backend builds a dictionary of function signatures to determine parameter types for interface casting at call sites.

**Should be:** Call nodes should already have resolved parameter types attached from inference, or lowering should emit `CastArg` wrappers.

---

## C#

### 1. Bool-to-Int Casting

**Location:** `csharp.py:1134-1190`, `1249`, `1343-1357`, `1420-1443`

```python
if _is_bool_type(left.typ):
    left_str = f"({left_str} ? 1 : 0)"
```

Multiple locations check `_is_bool_type()` and insert ternary conversion (`? 1 : 0`) when bool values are used in arithmetic, bitwise, shift, or comparison operations. This includes `BinaryOp`, `MinExpr`, `MaxExpr`, and builtin calls like `pow`, `divmod`, `abs`.

**Should be:** Inference or lowering should insert explicit `Cast(BOOL, INT)` nodes when bools flow into int-expecting operations.

---

### 2. String Comparison Detection

**Location:** `csharp.py:1203-1216`

```python
if _is_string_type(left.typ):
    if cs_op == "<":
        return f"string.Compare({left_str}, {right_str}) < 0"
```

The backend checks operand types to determine whether to use `string.Compare()` vs direct operators for string comparisons.

**Should be:** Lowering should emit `StringCompare` IR nodes for string ordering comparisons, distinct from primitive `BinaryOp`.

---

### 3. Truthy Type Dispatch

**Location:** `csharp.py:1113-1126`, `1224-1242`

```python
if _is_string_type(inner_type):
    return f"(!string.IsNullOrEmpty({inner_str}))"
if isinstance(inner_type, (Slice, Map, Set)):
    return f"({inner_str}.Count > 0)"
```

The backend switches on the inner type of `Truthy` and `UnaryOp("!")` nodes to emit type-appropriate truthiness/falsiness checks.

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `StringNonEmpty`, `SliceNonEmpty`, `IsNotNil`, etc.

---

### 4. Object Variable Tracking

**Location:** `csharp.py:222`, `497-498`, `527-528`, `554-556`, `1199-1202`, `1759-1762`

```python
self._object_vars: set[str] = set()
...
if cs_type == "object":
    self._object_vars.add(name)
```

The backend maintains `_object_vars` to track which variables were declared with `object` type, enabling correct cast emission when these variables are used in string comparisons.

**Should be:** IR `Var` nodes should carry their declared type vs narrowed type as separate fields, or lowering should emit explicit casts.

---

### 5. Interface Method Dispatch

**Location:** `csharp.py:231-246`, `1576-1586`

```python
self._method_to_interface: dict[str, str] = {}
for iface in module.interfaces:
    for m in iface.methods:
        self._method_to_interface[m.name] = iface.name
```

The backend builds a mapping from method names to interfaces, then uses it to cast `object`-typed receivers when calling interface methods.

**Should be:** Lowering should emit explicit `InterfaceCast` nodes when methods are called on variables that need interface narrowing, or `MethodCall` should carry the required interface type.

---

### 6. Substring Bounds Clamping

**Location:** `csharp.py:438-447`

```python
if func.name == "_substring":
    self._line("int clampedStart = Math.Max(0, Math.Min(start, len));")
    self._line("int clampedEnd = Math.Max(clampedStart, Math.Min(end, len));")
```

The backend special-cases `_substring` to emit clamping logic that matches Python slice semantics (out-of-bounds indices are clamped, not errors).

**Should be:** Lowering should emit `Substring` with explicit semantics flag (`clamp=True`), or emit the bounds-checking wrapper at lowering time.

---

### 7. Struct Literal Field Ordering

**Location:** `csharp.py:1696-1724`

```python
field_info = self.struct_fields.get(struct_name, [])
if field_info:
    ordered_args = []
    for field_name, field_type in field_info:
        if field_name in fields:
            ordered_args.append(self._expr(fields[field_name]))
```

The backend looks up struct field order from `_collect_struct_fields` to emit constructor arguments in the correct order for C# positional constructors.

**Should be:** `StructLit` IR node should already have fields in constructor parameter order, as determined by the fields phase.

---

### 8. Collection Containment Dispatch

**Location:** `csharp.py:1612-1623`

```python
if isinstance(container_type, Set):
    return f"{neg}{container_str}.Contains({item_str})"
if isinstance(container_type, Map):
    return f"{neg}{container_str}.ContainsKey({item_str})"
```

The backend switches on container type to emit the correct containment method (`.Contains()` vs `.ContainsKey()`).

**Should be:** Lowering should emit type-specific IR: `SetContains`, `MapContains`, `SliceContains`, `StringContains`.

---

### 9. IsNil vs IsEmpty Semantics

**Location:** `csharp.py:1277-1287`

```python
if isinstance(inner_type, (Slice, Map, Set)):
    inner_str = self._expr(inner)
    if negated:
        return f"({inner_str}.Count > 0)"
    return f"({inner_str}.Count == 0)"
```

For non-nullable collection types, `IsNil` is reinterpreted as an emptiness check (`.Count == 0`) since the collections can't actually be null in C#.

**Should be:** Lowering should distinguish `IsNil` (null check) from `IsEmpty` (collection emptiness), emitting the appropriate IR node based on type.

---

### 10. Function Value Boxing

**Location:** `csharp.py:232`, `242`, `1044-1045`

```python
self._function_names = {func.name for func in module.functions}
...
if name in self._function_names:
    return f"(Action){_safe_pascal(name)}"
```

The backend tracks module-level function names to wrap references in `(Action)` cast when functions are used as values (needed for C# delegate conversion).

**Should be:** `FuncRef` IR nodes should have an annotation indicating whether boxing/casting is needed when used as a value vs called directly.

---

### 11. BytesDecode Conversion

**Location:** `csharp.py:418-424`, `1369-1371`, `1509-1513`, `1649-1653`

```python
self._line("public static string _BytesToString(List<byte> bytes)")
...
if method == "decode":
    return f"{func_class}._BytesToString({obj_str})"
```

The backend emits a helper function `_BytesToString` and routes `bytes.decode()` calls and `str(bytes)` conversions to it.

**Should be:** Lowering should emit a `BytesDecode(bytes_expr, encoding)` IR node for bytes-to-string conversion.

---

### 12. Pop Return Value Handling

**Location:** `csharp.py:711-753`, `1482-1505`

```python
# Python pop() returns the removed element; C# RemoveAt is void
elem_type = self._element_type(receiver_type)
body = obj_str + ".RemoveAt(" + idx + "); return _tmp;"
return "((Func<" + elem_type + ", " + elem_type + ">)(_tmp => { " + body + " }))(" + obj_str + "[" + idx + "])"
```

The backend emits complex patterns (temp variables or lambdas) to handle `list.pop()` which returns the removed element in Python but is void in C#.

**Should be:** Lowering should emit a `ListPop(list, index)` IR node with explicit return-value semantics, rather than relying on backends to pattern-match `MethodCall("pop")`.

---

### 13. Tuple Pop Unpacking

**Location:** `csharp.py:563-566`, `711-753`

```python
case TupleAssign(targets=targets, value=value) if (
    isinstance(value, MethodCall) and value.method == "pop"
):
    self._emit_tuple_pop(stmt)
```

The backend pattern-matches `TupleAssign` with `MethodCall("pop")` to emit a special sequence: fetch element, remove from list, then unpack tuple fields.

**Should be:** Lowering should expand this pattern into explicit IR: `VarDecl` for temp, `ListRemove`, then `TupleAssign` from the temp variable.

---

## Dart

### 1. Bool-to-Int Casting

**Location:** `dart.py:1358-1390`, `1412-1433`, `1464-1474`, `1674-1693`

```python
if left_is_bool:
    left_str = f"({self._expr(left)} ? 1 : 0)"
```

Multiple locations check if operands are bool and insert ternary conversion (`? 1 : 0`) when bool values are used in arithmetic, bitwise, shift, or ordered comparison operations. This includes `BinaryOp`, `MinExpr`, `MaxExpr`, `UnaryOp`, and builtin calls like `pow`, `divmod`, `abs`.

**Should be:** Inference or lowering should insert explicit `Cast(BOOL, INT)` nodes when bools flow into int-expecting operations.

---

### 2. String Comparison Detection

**Location:** `dart.py:1396-1410`

```python
if op in (">=", "<=", ">", "<") and isinstance(left_type, Primitive) and left_type.kind == "string":
    return f"({left_str}.compareTo({right_str}) >= 0)"
```

The backend checks operand types to determine whether to use `.compareTo()` for string ordering comparisons.

**Should be:** Lowering should emit `StringCompare` IR nodes for string ordering comparisons, distinct from primitive `BinaryOp`.

---

### 3. Truthy Type Dispatch

**Location:** `dart.py:1330-1350`, `1445-1462`

```python
if _is_string_type(inner_type):
    return f"({inner_str}.isNotEmpty)"
if isinstance(inner_type, (Slice, Map, Set)):
    return f"({inner_str}.isNotEmpty)"
```

The backend switches on the inner type of `Truthy` and `UnaryOp("!")` nodes to emit type-appropriate truthiness/falsiness checks.

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `StringNonEmpty`, `SliceNonEmpty`, `IsNotNil`, etc.

---

### 4. Collection Containment Dispatch

**Location:** `dart.py:1824-1835`

```python
if isinstance(container_type, Set):
    return f"{neg}{container_str}.contains({item_str})"
if isinstance(container_type, Map):
    return f"{neg}{container_str}.containsKey({item_str})"
```

The backend switches on container type to emit the correct containment method.

**Should be:** Lowering should emit type-specific IR: `SetContains`, `MapContains`, `SliceContains`, `StringContains`.

---

### 5. IsNil vs IsEmpty Semantics

**Location:** `dart.py:1492-1501`

```python
if isinstance(inner_type, (Slice, Map, Set)):
    inner_str = self._expr(inner)
    if negated:
        return f"({inner_str}.isNotEmpty)"
    return f"({inner_str}.isEmpty)"
```

For non-nullable collection types, `IsNil` is reinterpreted as an emptiness check since the collections can't actually be null in Dart.

**Should be:** Lowering should distinguish `IsNil` (null check) from `IsEmpty` (collection emptiness), emitting the appropriate IR node based on type.

---

### 6. Substring Bounds Clamping

**Location:** `dart.py:1557-1568`, `1794-1822`, `2099-2108`

```python
if low and high:
    self._needed_helpers.add("_safeSubstring")
    return f"_safeSubstring({s_str}, {self._expr(low)}, {self._expr(high)})"
```

The backend emits helper functions (`_safeSubstring`) to clamp indices matching Python slice semantics where out-of-bounds indices are clamped, not errors.

**Should be:** Lowering should emit `Substring` with explicit semantics flag (`clamp=True`), or emit the bounds-checking wrapper at lowering time.

---

### 7. Struct Literal Field Ordering

**Location:** `dart.py:1924-1954`

```python
field_info = self.struct_fields.get(struct_name, [])
if field_info:
    ordered_args = []
    for field_name, field_type in field_info:
        if field_name in fields:
            ordered_args.append(self._expr(fields[field_name]))
```

The backend looks up struct field order from `_collect_struct_fields` to emit constructor arguments in the correct order for Dart positional constructors.

**Should be:** `StructLit` IR node should already have fields in constructor parameter order, as determined by the fields phase.

---

### 8. Nullable Return Type Detection

**Location:** `dart.py:549-591`, `604-606`, `631-634`

```python
def _has_null_return(self, stmts: list[Stmt]) -> bool:
    """Check if any statement returns null, directly or indirectly via variable."""
    ...
if func.body and self._has_null_return(func.body) and not isinstance(func.ret, Optional):
    ret = "dynamic"
```

The backend traverses function bodies looking for `return null` statements (including indirect returns via variables) to determine if the return type should be `dynamic` instead of the declared type.

**Should be:** A middleend pass should analyze return patterns and annotate functions with `may_return_null: bool`, or lowering should widen the return type when null returns are detected.

---

### 9. Interface Field Type Widening

**Location:** `dart.py:490-501`, `517-546`

```python
elif isinstance(fld.typ, InterfaceRef):
    # Interface fields use dynamic to allow null
    self._line(f"dynamic {name};")
...
elif isinstance(f.typ, InterfaceRef):
    # Use dynamic for interface types since Python allows None even without Optional
    param_parts.append(f"dynamic {_safe_name(f.name)}")
```

The backend uses `dynamic` for interface-typed fields and constructor params because Python code often passes None without Optional annotation.

**Should be:** Fields phase should detect when None is passed to non-Optional parameters and widen the type, or lowering should emit explicit `Optional(InterfaceRef)` when null flows are detected.

---

### 10. Type Switch Binding Renaming

**Location:** `dart.py:1002-1052`, `1246-1248`

```python
narrowed_name = f"{binding}{type_name}"
self._type_switch_binding_rename[stmt.binding] = narrowed_name
...
if name in self._type_switch_binding_rename:
    return self._type_switch_binding_rename[name]
```

The backend tracks binding renames for Dart's type pattern syntax (`case TypeName varName:`) which creates new bindings rather than narrowing the original variable.

**Should be:** Lowering should emit `TypeSwitch` with per-case binding names suitable for the target language, or a middleend pass should compute narrowed binding names.

---

### 11. Pop Return Value Handling

**Location:** `dart.py:783-786`, `952-984`

```python
case TupleAssign(targets=targets, value=value) if (
    isinstance(value, MethodCall) and value.method == "pop"
):
    self._emit_tuple_pop(stmt)
```

The backend pattern-matches `TupleAssign` with `MethodCall("pop")` to emit a special sequence: fetch element, remove from list, then unpack tuple fields.

**Should be:** Lowering should expand this pattern into explicit IR: `VarDecl` for temp, `ListRemove`, then `TupleAssign` from the temp variable.

---

### 12. Hoisted Variable Nullability

**Location:** `dart.py:655-687`

```python
def _is_nullable_reference_type(self, typ: Type) -> bool:
    ...
def _emit_hoisted_vars(self, stmt: ...) -> None:
    if self._is_nullable_reference_type(typ):
        self._line(f"dynamic {var_name};")
    else:
        dart_type = self._type(typ)
        default = self._default_value(typ)
        self._line(f"{dart_type} {var_name} = {default};")
```

The backend inspects hoisted variable types to determine whether to use `dynamic` (for reference types that may be null during control flow) or a typed declaration with default value.

**Should be:** The hoisting middleend pass should compute whether each hoisted variable needs nullable typing based on control flow analysis, attaching a `needs_nullable: bool` annotation.
