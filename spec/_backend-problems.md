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

---

## Java

### 1. Bool-to-Int Casting

**Location:** `java.py:1256-1261`, `1473-1534`, `1597-1646`, `1753-1755`, `1899-1921`, `2041-2047`, `2405-2409`

```python
def _java_coerce_bool_to_int(backend: "JavaBackend", expr: Expr) -> str:
    if _java_is_bool_in_java(expr):
        return f"({backend._expr(expr)} ? 1 : 0)"
    return backend._expr(expr)
```

Multiple locations check `_is_bool_type()` or `_java_is_bool_in_java()` and insert ternary conversion (`? 1 : 0`) when bool values are used in arithmetic, bitwise, shift, comparison, or power operations. This includes `BinaryOp`, `MinExpr`, `MaxExpr`, `UnaryOp`, and builtin calls like `pow`, `divmod`, `abs`, `min`, `max`.

**Should be:** Inference or lowering should insert explicit `Cast(BOOL, INT)` nodes when bools flow into int-expecting operations.

---

### 2. String Comparison Detection

**Location:** `java.py:1703-1715`

```python
if java_op == "==" and _is_string_type(left.typ):
    return f"{left_str}.equals({right_str})"
if java_op == "<" and _is_string_type(left.typ):
    return f"({left_str}.compareTo({right_str}) < 0)"
```

The backend checks operand types to determine whether to use `.equals()` vs `==` for string equality and `.compareTo()` for ordering comparisons.

**Should be:** Lowering should emit `StringCompare` IR nodes for string comparisons, distinct from primitive `BinaryOp`.

---

### 3. Truthy Type Dispatch

**Location:** `java.py:1559-1573`

```python
if _is_string_type(inner_type) or isinstance(inner_type, (Slice, Map, Set)):
    return f"(!{inner_str}.isEmpty())"
if inner_type == Primitive(kind="int"):
    return f"({inner_str} != 0)"
return f"({inner_str} != null)"
```

The backend switches on the inner type of `Truthy` nodes to emit type-appropriate truthiness checks.

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `StringNonEmpty`, `SliceNonEmpty`, `IsNotNil`, etc.

---

### 4. Collection Containment Dispatch

**Location:** `java.py:2016-2033`

```python
if isinstance(container_type, Set):
    return f"{neg}{container_str}.contains({item_str})"
if isinstance(container_type, Map):
    return f"{neg}{container_str}.containsKey({item_str})"
if isinstance(container_type, Primitive) and container_type.kind == "string":
    return f"{container_str}.indexOf({item_str}) != -1"
```

The backend switches on container type to emit the correct containment method.

**Should be:** Lowering should emit type-specific IR: `SetContains`, `MapContains`, `SliceContains`, `StringContains`.

---

### 5. Struct Literal Field Ordering

**Location:** `java.py:1851-1877`

```python
field_info = self.struct_fields.get(struct_name, [])
if field_info:
    ordered_args = []
    for field_name, field_type in field_info:
        if field_name in fields:
            ordered_args.append(self._expr(fields[field_name]))
```

The backend looks up struct field order from `_collect_struct_fields` to emit constructor arguments in the correct order for Java positional constructors.

**Should be:** `StructLit` IR node should already have fields in constructor parameter order, as determined by the fields phase.

---

### 6. Interface Method Dispatch

**Location:** `java.py:502-505`, `1986-1994`

```python
self._method_to_interface: dict[str, str] = {}
for iface in module.interfaces:
    for m in iface.methods:
        self._method_to_interface[to_camel(m.name)] = iface.name
...
if self._type(obj.typ) == "Object":
    if camel_method in self._method_to_interface:
        iface_name = self._method_to_interface[camel_method]
        return f"(({_java_safe_class(iface_name)}) {obj_str}).{camel_method}({args_str})"
```

The backend builds a mapping from method names to interfaces, then uses it to cast `Object`-typed receivers when calling interface methods.

**Should be:** Lowering should emit explicit `InterfaceCast` nodes when methods are called on variables that need interface narrowing, or `MethodCall` should carry the required interface type.

---

### 7. Substring Bounds Clamping

**Location:** `java.py:729-738`

```python
if func.name == "_substring":
    self._line(f"static {ret} {name}({params}) {{")
    self._line("int clampedStart = Math.max(0, Math.min(start, len));")
    self._line("int clampedEnd = Math.max(clampedStart, Math.min(end, len));")
    self._line("return s.substring(clampedStart, clampedEnd);")
```

The backend special-cases `_substring` to emit clamping logic that matches Python slice semantics (out-of-bounds indices are clamped, not errors).

**Should be:** Lowering should emit `Substring` with explicit semantics flag (`clamp=True`), or emit the bounds-checking wrapper at lowering time.

---

### 8. Tuple Type Collection

**Location:** `java.py:259-460` (`_java_register_tuple`, `_java_visit_type`, `_java_visit_expr`, `_java_visit_stmt`)

The backend manually traverses the entire IR module to collect all tuple types, building a dictionary of tuple signatures to record names for emission. This requires ~200 lines of visitor code duplicating AST traversal logic.

**Should be:** A middleend pass that computes `Module.used_tuple_types` as a set of tuple signatures.

---

### 9. Pop Return Value Handling

**Location:** `java.py:834-838`, `1213-1254`

```python
case TupleAssign(targets=targets, value=value) if (
    isinstance(value, MethodCall) and value.method == "pop"
):
    self._emit_tuple_pop(stmt)
```

The backend pattern-matches `TupleAssign` with `MethodCall("pop")` to emit a special sequence: fetch element, remove from list, then unpack tuple fields. Java's `List.remove()` returns the removed element, but this requires temp variable handling.

**Should be:** Lowering should emit a `ListPop(list, index)` IR node with explicit return-value semantics, or expand the pattern into `VarDecl` + `ListRemove` + `TupleAssign` from temp.

---

### 10. Type Switch Binding Renaming

**Location:** `java.py:1008-1022`

```python
narrowed_name = f"{bind_name}{type_name.replace('Node', '')}"
self._type_switch_binding_rename[binding] = narrowed_name
...
self._line(f"{keyword} ({bind_name} instanceof {type_name} {narrowed_name}) {{")
```

The backend tracks binding renames for Java 16+ pattern matching syntax (`instanceof TypeName varName`) which creates new bindings rather than narrowing the original variable.

**Should be:** Lowering should emit `TypeSwitch` with per-case binding names suitable for the target language, or a middleend pass should compute narrowed binding names.

---

### 11. BytesDecode/StringToBytes Conversion

**Location:** `java.py:699-707`, `1419-1423`, `1964-1966`, `2062-2088`

```python
if method == "decode":
    self._needs_bytes_helper = True
    return f"ParableFunctions._bytesToString({obj_str})"
```

The backend emits helper functions `_bytesToString` and `_stringToBytes` and routes `bytes.decode()` calls and `str(bytes)` conversions to them.

**Should be:** Lowering should emit `BytesDecode(bytes_expr, encoding)` and `StringEncode(str_expr, encoding)` IR nodes for bytes/string conversion.

---

### 12. Char vs String Single-Character Optimization

**Location:** `java.py:1673-1691`

```python
if java_op in ("==", "!=") and isinstance(right, StringLit) and len(right.value) == 1:
    inner_left = left.expr if isinstance(left, Cast) else left
    if isinstance(inner_left, Index):
        obj_type = inner_left.obj.typ
        if isinstance(obj_type, Primitive) and obj_type.kind == "string":
            return f"{obj_str}.charAt({idx_str}) == {char_lit}"
```

The backend detects comparisons between string indexing and single-character string literals, emitting character literal comparisons (`charAt(i) == 'c'`) instead of string comparisons.

**Should be:** Inference should normalize single-char string comparisons to char comparisons, or lowering should emit `CharCompare` nodes.

---

### Uncompensated: Char vs String Representation

The dominant uncompensated issue is Python's single-type `str` vs Java's `char`/`String` duality. Python has only strings; Java distinguishes `char` (primitive) from `String` (object). The backend now detects single-char string comparisons but many patterns remain:

**Consequences:**
- ~175 remaining `String.valueOf()` calls for character-to-string contexts
- Helper predicates take String instead of char: `_isWhitespace(String)` should be `Character.isWhitespace(char)` (~356 helper call sites)
- `CharClassify` uses string-based pattern instead of direct char: `s.chars().allMatch(Character::isDigit)` instead of `Character.isDigit(c)`

**Should be:** Frontend/middleend should track character vs string semantics, emitting `char` type for single-character literals and `charAt()` results, with primitive `==` for char comparisons.

---

## Go

### 1. Hoisted Variable Tuple Type Inference

**Location:** `go.py:1099`, `2464-2471`

```python
ret_type = self._infer_tuple_element_type(name, stmt, self._current_return_type)
...
def _infer_tuple_element_type(self, var_name: str, stmt: If, ret_type: Tuple) -> Type | None:
    """Infer which tuple element a variable corresponds to by scanning returns."""
    pos = _scan_for_return_position(stmt.then_body, var_name)
```

The backend scans return statements to infer hoisted variable types when the function returns a tuple. It matches variable names in `return` tuple literals to determine their position and type.

**Should be:** The middleend hoisting pass should annotate `hoisted_vars` with complete type information, never leaving `typ=None` for variables that can be inferred from return position analysis.

---

### 2. String Character Indexing Helpers

**Location:** `go.py:567-612`, `1762-1781`, `2270`

```python
("_runeAt(", """func _runeAt(s string, i int) string { ... }"""),
("_runeLen(", """func _runeLen(s string) int { ... }"""),
("_Substring(", """func _Substring(s string, start int, end int) string { ... }"""),
...
return f"_runeAt({obj}, {idx})"
```

Python strings are character sequences; Go strings are byte sequences. The backend emits `_runeAt`, `_runeLen`, and `_Substring` helper calls for all string indexing/slicing operations to match Python semantics.

**Should be:** Frontend should emit distinct IR nodes for character-based string operations (`CharAt`, `CharLen`, `Substring`) vs byte-based operations. Currently all string indexing emits `Index` nodes without semantic distinction.

---

### 3. Interface Nil Check Reflection

**Location:** `go.py:557-564`, `2174-2186`

```python
("_isNilInterfaceRef(", """func _isNilInterfaceRef(i interface{}) bool {
    if i == nil { return true }
    v := reflect.ValueOf(i)
    return v.Kind() == reflect.Ptr && v.IsNil()
}"""),
...
if isinstance(expr.expr.typ, InterfaceRef):
    return f"_isNilInterfaceRef({inner})"
```

Go requires reflection to correctly check if an interface value contains a typed nil pointer. The backend conservatively emits `_isNilInterfaceRef()` for all interface nil checks.

**Should be:** Middleend could track when expressions are definitely `interface{}` vs typed nil pointers, allowing direct `== nil` comparison in simple cases. Many nil checks are on freshly assigned interface values where the concrete type is known.

---

### 4. Type Switch Binding Name Extraction

**Location:** `go.py:1482-1485`, `1544`, `1560-1568`

```python
narrowed_name = f"{binding}{self._extract_type_suffix(go_type)}"
...
def _extract_type_suffix(self, go_type: str) -> str:
    for prefix in ("Arith", "Cond", ""):
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix):]
```

The backend uses hardcoded prefixes ("Arith", "Cond") to extract type suffixes for type switch binding names. These are Parable-specific naming conventions for arithmetic/conditional expression types.

**Should be:** Frontend should emit type-agnostic IR for narrowed bindings. `TypeSwitch` cases should carry explicit narrowed binding names rather than relying on backend string manipulation.

---

### 5. IIFE Ternary Expansion

**Location:** `go.py:2124-2137`

```python
def _emit_expr_Ternary(self, expr: Ternary) -> str:
    # Go doesn't have ternary, emit as IIFE
    ...
    return f"func() {go_type} {{ if {cond} {{ return {then_expr} }} else {{ return {else_expr} }} }}()"
```

Go lacks a ternary operator. The backend emits immediately-invoked function expressions (IIFEs) for all `Ternary` nodes.

**Should be:** Frontend could emit `Ternary` with a flag indicating if/else expansion is acceptable, or middleend could lift ternaries to variable assignments when used in statement context. The IIFE pattern is non-idiomatic and adds runtime overhead.

---

### 6. ParseInt Helper Wrapper

**Location:** `go.py:534-538`, `2350`

```python
("_parseInt(", """func _parseInt(s string, base int) int {
    n, _ := strconv.ParseInt(s, base, 64)
    return int(n)
}"""),
...
return f"_parseInt({self._emit_expr(expr.string)}, {self._emit_expr(expr.base)})"
```

The backend emits a helper that silently ignores parse errors, matching Python's `int()` behavior. Go's `strconv.ParseInt` returns `(int64, error)`.

**Should be:** Lowering should emit a `ParseInt` node with explicit error-handling semantics (ignore, panic, or return optional), letting backends choose appropriate inline patterns vs helpers.

---

### 7. Integer Pointer Helper

**Location:** `go.py:495-501`

```python
("_intPtr(", """func _intPtr(val int) *int {
    if val == -1 { return nil }
    return &val
}"""),
```

The backend emits a helper to convert sentinel integer values (-1) to nil pointers for `Optional[int]` fields.

**Should be:** Lowering should emit explicit `SentinelToOptional(expr, sentinel_value)` nodes. The sentinel value (-1) should be determined by type analysis in middleend, not hardcoded in backend helpers.

---

### 8. Truthy Type Dispatch

**Location:** `go.py:2188-2202`

```python
def _emit_expr_Truthy(self, expr: Truthy) -> str:
    inner_type = expr.expr.typ
    if inner_type == STRING:
        return f"(len({inner}) > 0)"
    if isinstance(inner_type, (Slice, Map, Set)):
        return f"(len({inner}) > 0)"
    return f"({inner} != nil)"
```

The backend switches on the inner type of `Truthy` nodes to emit type-appropriate truthiness checks.

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `StringNonEmpty`, `SliceNonEmpty`, `IsNotNil`, etc.

---

### Uncompensated: String vs []rune Representation

The dominant uncompensated issue is Go's string/rune duality. Python strings are character sequences; Go strings are UTF-8 byte sequences. The backend defensively emits `_runeAt`/`_runeLen`/`_Substring` helpers for all string operations, but idiomatic Go would:

1. Analyze string usage patterns in frontend/middleend
2. Type variables as `[]rune` when indexed by character (e.g., lexer source)
3. Convert once at scope entry: `runes := []rune(source)`
4. Use direct indexing: `runes[i]` instead of `_runeAt(source, i)`

**Consequences:**
- ~100 while-style loops (`for i < n`) instead of `for i, c := range runes`
- ~60 helper predicates take `string` instead of `rune`: `_strIsDigit(string)` should be `unicode.IsDigit(rune)` with direct rune comparisons

**Should be:** Frontend/middleend should analyze string access patterns and emit `[]rune` type for variables that are character-indexed, with explicit `StringToRunes` conversion at declaration.

---

## JavaScript / TypeScript

The JS and TS backends share a common base in `jslike.py`. TypeScript extends JavaScript with type annotations but inherits all the compensations.

### 1. Bool-to-Int Casting

**Location:** `jslike.py:1159-1166` (MinExpr/MaxExpr), `jslike.py:987-994` (pow base/exp)

```python
l = f"({self._expr(left)} ? 1 : 0)" if left.typ == BOOL else self._expr(left)
r = f"({self._expr(right)} ? 1 : 0)" if right.typ == BOOL else self._expr(right)
return f"Math.min({l}, {r})"
```

Multiple locations check if operands are `BOOL` and insert ternary conversion (`? 1 : 0`) when bool values flow into `Math.min`, `Math.max`, or power operations.

**Should be:** Inference or lowering should insert explicit `Cast(BOOL, INT)` nodes when bools flow into int-expecting operations.

---

### 2. Truthy Type Dispatch

**Location:** `jslike.py:1621-1638` (`_truthy_expr`)

```python
if isinstance(inner_type, Map) or _is_set_expr(e):
    return f"({inner_str}.size > 0)"
if isinstance(inner_type, (Slice, Tuple, Array)) or inner_type == STRING:
    return f"({inner_str}.length > 0)"
```

The backend switches on the inner type of `Truthy` nodes to emit type-appropriate truthiness checks (`.size > 0`, `.length > 0`, `!= null`).

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `StringNonEmpty`, `SliceNonEmpty`, `MapNonEmpty`, `IsNotNil`, etc.

---

### 3. Map Key Type Coercion (Python Key Equivalence)

**Location:** `jslike.py:1290-1328` (`_coerce_map_key`), `jslike.py:7-9` (acknowledged limitation in docstring)

```python
# BOOL variable → INT
if map_key == "int" and key_typ == "bool":
    return f"({key_code} ? 1 : 0)"
# FLOAT variable → INT
if map_key == "int" and key_typ == "float":
    return f"Math.trunc({key_code})"
```

Python treats `True==1` and `False==0` as equivalent dict keys. JS `Map` treats them as different. The backend coerces keys at runtime based on declared map key type. The docstring explicitly notes this only works in VarDecl initializers and direct Index/method access—not in Assign, function args, return statements, or comprehensions.

**Should be:** Inference should track Python's key equivalence semantics, and lowering should emit explicit `CoerceMapKey(expr, target_key_type)` nodes when key types don't match the map's declared key type.

---

### 4. String Length Character Counting

**Location:** `jslike.py:1922-1925` (`_len_expr`)

```python
if inner_type == STRING:
    return f"[...{self._expr(inner)}].length"
```

Python's `len()` on strings counts Unicode code points. JavaScript's `.length` counts UTF-16 code units (emoji are 2). The backend spreads strings into arrays to count correctly.

**Should be:** Lowering should emit `CharLen(string)` IR nodes for character-based length, distinct from byte-based `Len()`. The spec mentions `CharLen` but the backend is compensating for its incomplete use.

---

### 5. Python Modulo Semantics

**Location:** `jslike.py:1831-1838`

```python
if op == "%":
    if self._is_known_non_negative(left) and self._is_known_non_negative(right):
        return f"{left_str} % {right_str}"
    return f"(({left_str} % {right_str}) + {right_str}) % {right_str}"
```

Python modulo returns a result with the sign of the divisor; JS modulo returns a result with the sign of the dividend. The backend wraps modulo with `((a % b) + b) % b` unless it can prove both operands are non-negative.

**Should be:** Lowering should emit `PythonMod` vs `CMod` IR nodes, or a middleend pass should annotate expressions with `is_known_non_negative` based on dataflow analysis.

---

### 6. Collection Containment Dispatch

**Location:** `jslike.py:1996-2017` (`_containment_check`)

```python
if isinstance(container_type, Set):
    return f"{neg}{container_str}.has({item_str})"
if isinstance(container_type, Map):
    coerced_key = self._coerce_map_key(container_type.key, item)
    return f"{neg}{container_str}.has({coerced_key})"
if is_bytes_type(container_type):
    return f"{neg}arrContains({container_str}, {item_str})"
```

The backend switches on container type to emit the correct containment method (`.has()`, `.includes()`, `arrContains()`).

**Should be:** Lowering should emit type-specific IR: `SetContains`, `MapContains`, `SliceContains`, `StringContains`, `BytesContains`.

---

### 7. Set/Map Operations with Tuple Elements

**Location:** `jslike.py:1341-1350` (tuple-key map index), `jslike.py:1756-1791` (tuple-element set binary ops), `jslike.py:2003-2008` (tuple containment)

```python
if isinstance(obj_type.key, Tuple):
    return f"tupleMapGet({obj_str}, {self._expr(index)})"
...
if _is_tuple_set_expr(left) or _is_tuple_set_expr(right):
    return f"[...{left_str}].every(x => tupleSetHas({right_str}, x))"
```

JS `Set` and `Map` use reference equality; Python uses value equality. The backend emits helper function calls (`tupleSetHas`, `tupleSetAdd`, `tupleMapGet`, `tupleMapHas`) when sets/maps contain tuple elements/keys.

**Should be:** Inference should annotate sets/maps containing non-primitive types with `needs_value_equality: true`, and lowering should emit distinct IR nodes (`TupleSetContains`, `TupleMapGet`) or generate the helper calls during lowering.

---

### 8. Tuple vs List Mutation Semantics

**Location:** `jslike.py:383-395` (OpAssign handling)

```python
if isinstance(value.typ, Tuple) or isinstance(target_type, Tuple):
    self._line(f"{lv} = [...{lv}, ...{val}];")
else:
    self._line(f"{lv}.push(...{val});")
```

Python tuples are immutable; `+=` creates a new tuple. Python lists are mutable; `+=` modifies in place. The backend checks if the target is a `Tuple` type to use concatenation instead of `.push()`.

**Should be:** Lowering should distinguish tuple vs list augmented assignment, emitting `TupleConcat` (returns new tuple) vs `ListExtend` (mutates in place) IR nodes.

---

### 9. Dict View Set Operations

**Location:** `jslike.py:1693-1704` (`_set_operand`), `jslike.py:1770-1807`

```python
if _is_dict_items_view(expr):
    return f"(function() {{ const s = new Set(); for (const t of {expr_str}) tupleSetAdd(s, t); return s; }})()"
if _is_dict_view_expr(expr):
    return f"new Set({expr_str})"
```

Python dict views (`.keys()`, `.values()`, `.items()`) support set operations. The backend detects view expressions and wraps them appropriately for set operators.

**Should be:** Lowering should convert dict view expressions to explicit set types when used in set operations, or emit `DictKeysView`, `DictValuesView`, `DictItemsView` IR nodes with set-compatible interfaces.

---

### 10. Hoisted Variable Tracking (JS only)

**Location:** `javascript.py:499-504` (`_hoisted_vars_hook`), `javascript.py:637-655` (`_get_hoisted_vars`)

```python
def _hoisted_vars_hook(self, stmt: Stmt) -> None:
    hoisted = _get_hoisted_vars(stmt)
    for name, _ in hoisted:
        js_name = _camel(name)
        if name not in self._hoisted_vars:
            self._line(f"var {js_name};")
            self._hoisted_vars.add(name)
```

JavaScript's block scoping with `let` differs from Python's function scoping. The JS backend tracks hoisted variables and emits `var` declarations at function scope, reading the middleend's `hoisted_vars` annotation but doing additional bookkeeping.

**Should be:** The middleend hoisting pass should compute complete hoisting requirements. The backend should only need to read `stmt.hoisted_vars` and emit declarations, without tracking state in `_hoisted_vars`.

---

### 11. Implicit Return Null for Void Functions

**Location:** `javascript.py:494-497` (`_post_function_body`), `typescript.py:584-586`

```python
def _post_function_body(self, func: Function) -> None:
    if _is_void_func(func):
        self._line("return null;")
```

Python functions return `None` implicitly. The backend checks if a function returns `VOID` and has no explicit `Return` statement, adding `return null;` for JavaScript.

**Should be:** Lowering should add explicit `Return(NilLit)` at the end of void functions when there's no explicit return, or middleend should add a `needs_implicit_return: bool` annotation.

---

### 12. Array/Map/Set Comparison

**Location:** `jslike.py:1739-1768` (array comparison), `jslike.py:1809-1822` (map comparison)

```python
if _is_array_type(left.typ) or _is_array_type(right.typ):
    if op == "==":
        return f"arrEq({left_str}, {right_str})"
if _is_map_expr(left) and _is_map_expr(right):
    if op == "==":
        return f"mapEq({left_str}, {right_str})"
```

Python uses value equality for collections; JS uses reference equality. The backend emits helper function calls (`arrEq`, `mapEq`) for collection comparisons.

**Should be:** Lowering should emit `ArrayEquals`, `MapEquals`, `SetEquals` IR nodes for collection comparisons, distinct from `BinaryOp("==")` for primitives.

---

### 13. String Split Semantics

**Location:** `jslike.py:1549-1562`

```python
if receiver_type == STRING and method == "split" and len(args) == 0:
    return f"{self._expr(obj)}.trim().split(/\\s+/).filter(Boolean)"
if receiver_type == STRING and method == "split" and len(args) == 2:
    # Python splits at most maxsplit times
    return f"((m = {maxsplit}) === 0 ? [{obj_str}] : ...)"
```

Python `str.split()` with no args splits on whitespace and removes empties. Python `str.split(sep, maxsplit)` limits splits. JS `.split()` differs in both behaviors.

**Should be:** Lowering should emit `StringSplitWhitespace()` for no-arg split and `StringSplit(sep, maxsplit)` with explicit semantics, rather than relying on backends to pattern-match `MethodCall("split")`.

---

### Uncompensated: BigInt for Large Integers

**Location:** `jslike.py:1281-1282`

```python
if abs(value) > 9007199254740991:
    return f"{value}n"
```

The backend emits BigInt literals for integers exceeding JS's safe integer range. However, operations mixing BigInt and Number fail at runtime. The backend doesn't track BigInt contamination through expressions.

**Should be:** Inference should track which expressions may exceed safe integer range and emit `BigInt` type annotations, with explicit `BigIntToNumber` / `NumberToBigInt` conversions at boundaries. Currently only literal detection is implemented.

---

## Lua

The Lua backend targets Lua 5.4+ with some compatibility considerations for Lua 5.1. Key language differences include 1-based indexing, classes via metatables, and no native `continue` statement.

### 1. 1-Based Indexing Adjustment

**Location:** `lua.py:1157-1168` (Index), `lua.py:1787-1803` (SliceExpr), `lua.py:1844-1853` (IndexLV lvalue), `lua.py:919-932` (ForRange)

```python
if isinstance(obj_type, (Slice, Array)):
    return f"{obj_str}[{idx_str} + 1]"
...
# In ForRange, convert to 0-based index
self._line(f"{idx} = {idx} - 1")
```

The backend adds `+ 1` to all array/slice/string indices at access time to convert from Python's 0-based to Lua's 1-based indexing. For enumeration loops, it also subtracts 1 from the index variable to return to Python semantics.

**Should be:** Lowering should emit explicit `Index1Based(obj, idx)` IR nodes or add an `index_base: int` annotation to Index nodes, rather than having every backend handle the adjustment independently.

---

### 2. Bool-to-Int Conversion

**Location:** `lua.py:1203-1243` (Call handling for min/max/abs/pow/divmod), `lua.py:1317-1451` (BinaryOp arithmetic/comparison/bitwise/shift), `lua.py:1456-1461` (UnaryOp), `lua.py:1587-1594` (MinExpr/MaxExpr), `lua.py:1895-1909` (`_bool_to_int` helper)

```python
# Lua bools don't support arithmetic
if op in ("+", "-", "*", "/", "%", "//") and (left_is_bool or right_is_bool):
    left_str = f"({self._expr(left)} and 1 or 0)" if left_is_bool else ...
```

Extensive code checks if operands have `BOOL` type and wraps them with `(expr and 1 or 0)`. This pattern appears in:
- Arithmetic operators (+, -, *, /, %, //)
- Comparison operators (<, >, <=, >=, ==, !=)
- Bitwise operators (&, |, ^)
- Shift operators (<<, >>)
- Unary operators (-, ~)
- Built-in functions (min, max, abs, pow, divmod)

**Should be:** Inference should insert explicit `Cast(BOOL, INT)` nodes when bools flow into arithmetic, bitwise, or comparison operations that require numeric operands.

---

### 3. Truthy Type Dispatch

**Location:** `lua.py:1277-1304` (Truthy handling)

```python
if isinstance(inner_type, Slice):
    return f"(#({expr_str}) > 0)"
if isinstance(inner_type, (Map, Set)):
    return f"(next({expr_str}) ~= nil)"
if inner_type == Primitive(kind="int"):
    return f"({expr_str} ~= 0)"
```

Backend switches on the inner type of `Truthy` nodes to emit type-appropriate truthiness checks. Lua's truthiness semantics (only `nil` and `false` are falsy) differ from Python's.

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `SliceNonEmpty`, `MapNonEmpty`, `IntNonZero`, etc.

---

### 4. Containment Check Dispatch

**Location:** `lua.py:1618-1632` (`_containment_check`)

```python
if isinstance(container_type, Set):
    self._needed_helpers.add("_set_contains")
    return f"{neg}_set_contains({container_str}, {item_str})"
if isinstance(container_type, Map):
    return f"({neg}({container_str}[{item_str}] ~= nil))"
if isinstance(container_type, Primitive) and container_type.kind == "string":
    return f"({neg}(string.find({container_str}, {item_str}, 1, true) ~= nil))"
# Array/Slice - need to search
return f"({neg}(function() for _, v in ipairs(...) ... end)())"
```

Backend switches on container type to emit correct containment semantics. Sets use a helper, maps use `[key] ~= nil`, strings use `string.find`, and arrays require an IIFE loop.

**Should be:** Lowering should emit type-specific IR: `SetContains`, `MapContains`, `StringContains`, `SliceContains`.

---

### 5. Hoisted Variable Computation

**Location:** `lua.py:349-395` (`_collect_assigned_vars`), `lua.py:620-641` (`_emit_function`), `lua.py:643-669` (`_emit_method`)

```python
def _emit_function(self, func: Function) -> None:
    all_vars = self._collect_assigned_vars(func.body)
    param_names = {p.name for p in func.params}
    local_vars = all_vars - param_names - {"_"}
    self._hoisted_vars = set(local_vars)
    ...
    if local_vars:
        self._line(f"local {', '.join(_safe_name(v) for v in sorted(local_vars))}")
```

Backend traverses the entire function body recursively to collect all assigned variable names, then emits `local` declarations at function start. Lua requires variables to be declared before use.

**Should be:** Middleend hoisting pass should compute `func.hoisted_vars`. Backend should simply read this annotation instead of recomputing it.

---

### 6. Continue Statement Transformation

**Location:** `lua.py:224-270` (`_scan_for_continue`), `lua.py:236-310` (`_body_has_continue`, `_body_has_direct_continue`), `lua.py:772-774`, `937-938`, `960-961`, `976-977` (label emission)

```python
case Continue(label=_):
    self._line("goto continue")
...
if has_continue:
    self._line("::continue::")
```

Backend scans all function bodies for `continue` statements and emits `goto continue` with `::continue::` labels at loop ends. Lua 5.1 doesn't have `continue`; Lua 5.2+ supports `goto`.

**Should be:** A middleend pass should annotate loops with `has_continue: bool`, or transform loops containing `continue` into a form that doesn't require labels. Backend shouldn't need to scan for continue statements.

---

### 7. Try-Catch Return Propagation

**Location:** `lua.py:983-1065` (`_emit_try_catch`)

```python
has_return = self._body_has_return(body)
self._line("local _ok, _err = pcall(function()")
...
if has_return:
    self._line("if _ok then return _err end")
```

Backend wraps try body in `pcall(function() ... end)`, scans for return statements, and emits `if _ok then return _err end` to propagate successful returns. This requires analyzing the try body at emission time.

**Should be:** Lowering could emit a more explicit `TryCatchReturn(body, catches)` node, or middleend should annotate TryCatch with `body_has_return: bool`.

---

### 8. Ternary Falsy Guard

**Location:** `lua.py:1482-1494` (Ternary), `lua.py:1598-1608` (`_could_be_falsy`)

```python
if self._could_be_falsy(then_expr):
    return f"(function() if {cond_str} then return {then_str} else return {else_str} end end)()"
return f"({cond_str} and {then_str} or {else_str})"
```

Lua's `cond and a or b` idiom fails when `a` evaluates to `false` or `nil`. Backend checks if the then-branch could be falsy (boolean false, nil, or optional type) and emits a function wrapper instead.

**Should be:** Inference should annotate expressions with `could_be_falsy: bool`, or lowering should emit distinct IR: `SafeTernary(cond, then, else)` when then-branch needs protection.

---

### 9. Map/Set Length Calculation

**Location:** `lua.py:1531-1537` (Len)

```python
if isinstance(inner_type, (Map, Set)):
    inner_str = self._expr(inner)
    return f"(function() local c = 0; for _ in pairs({inner_str}) do c = c + 1 end; return c end)()"
```

Lua's `#` operator only counts consecutive integer keys starting from 1. For maps/sets (which use arbitrary keys), backend emits an IIFE that iterates with `pairs()` and counts.

**Should be:** Lowering should emit `MapLen(expr)`, `SetLen(expr)` IR nodes distinct from `Len(expr)` for arrays/slices.

---

### 10. String Concatenation Operator

**Location:** `lua.py:698-703` (OpAssign), `lua.py:1310-1316` (BinaryOp)

```python
if op == "+" and (self._is_string_type(left.typ) or self._is_string_type(right.typ)):
    return f"{left_str} .. {right_str}"
```

Backend detects string types in binary `+` operations and changes the operator to `..` (Lua's string concatenation operator).

**Should be:** Lowering should emit `StringConcat(parts)` IR for all string concatenation, not `BinaryOp("+", str, str)`.

---

### 11. Method Call Type Dispatch

**Location:** `lua.py:1634-1785` (`_method_call`)

The method implements a large dispatch table based on receiver type:
- String methods: join, split, lower, upper, find, rfind, startswith, endswith, replace, strip, etc.
- Slice methods: append, extend, copy, pop
- Map methods: get
- Set methods: add, contains

Each method has type-specific Lua implementations.

**Should be:** Lowering should emit type-specific method IR: `StringJoin`, `StringSplit`, `SliceAppend`, `SlicePop`, `MapGet`, `SetAdd`, etc.

---

### 12. Expression Statement Parenthesis Guard

**Location:** `lua.py:707-715` (ExprStmt)

```python
if e.startswith("(") and self._needs_paren_guard:
    self._line(";" + e)
else:
    self._line(e)
self._needs_paren_guard = True
```

Lua has an ambiguous grammar: `f()\n(g)()` could be parsed as `f()(g)()` (calling f's result with g as argument). Backend tracks when the previous statement could be misinterpreted and prefixes expression statements starting with `(` with `;`.

**Should be:** This is a Lua-specific syntax quirk. A middleend pass could mark statements needing guards, but this is arguably acceptable as backend-only logic.

---

### 13. Right Shift Semantics

**Location:** `lua.py:1434-1451`

```python
if op == ">>":
    if left_is_bool or (isinstance(left, IntLit) and left.value >= 0):
        return f"{left_for_shift} >> {right_for_shift}"
    return f"{left_str} // (1 << {right_str})"
```

Lua's `>>` is a logical right shift (fills with zeros). Python's `>>` is arithmetic (preserves sign). Backend detects negative operands and emits `// (1 << n)` for correct arithmetic shift semantics.

**Should be:** Lowering should emit distinct `ArithmeticRightShift` vs `LogicalRightShift` IR nodes, or inference should annotate with `is_known_non_negative`.
