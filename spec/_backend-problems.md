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

---

## Perl

The Perl backend targets Perl 5.36+ with sigil-based variables, duck typing, and `eval {}` for exception handling.

### 1. Bool-to-Int Casting

**Location:** `perl.py:1483-1517` (BinaryOp), `perl.py:1540-1542` (UnaryOp), `perl.py:1256-1301` (Call for pow/abs/divmod)

```python
if op in ("+", "-", "*", "/", "%", "//", "&", "|", "^", "<<", ">>", "<", ">", "<=", ">="):
    if left_is_bool or right_is_bool:
        left_str = self._bool_to_int(left) if _perl_needs_bool_coerce(left) else ...
```

Multiple locations check if operands are bool and wrap with `(expr ? 1 : 0)`. This includes arithmetic, comparison, bitwise, and shift operations, as well as builtin calls (`pow`, `divmod`, `abs`).

**Should be:** Inference or lowering should insert explicit `Cast(BOOL, INT)` nodes when bools flow into int-expecting operations.

---

### 2. Truthy Type Dispatch

**Location:** `perl.py:1454-1472` (`Truthy` handling)

```python
if isinstance(inner_type, (Slice, Map, Set)):
    if isinstance(inner_type, Map):
        return f"(scalar(keys %{{({self._expr(e)} // {{}})}}) > 0)"
    return f"(scalar(@{{({self._expr(e)} // [])}}) > 0)"
if inner_type == Primitive(kind="string"):
    return f"(length({self._expr(e)}) > 0)"
return f"({self._expr(e)} ? 1 : 0)"
```

Backend switches on the inner type of `Truthy` nodes to emit type-appropriate truthiness checks. Perl's truthiness (0, "", and undef are false) differs from Python's.

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `StringNonEmpty`, `SliceNonEmpty`, `MapNonEmpty`, `IsNotNil`, etc.

---

### 3. String Operator Detection

**Location:** `perl.py:1871-1898` (`_binary_op`)

```python
is_string = _is_string_type(left_type) or (right_type is not None and _is_string_type(right_type))
match op:
    case "==" if is_string: return "eq"
    case "!=" if is_string: return "ne"
    case "<" if is_string: return "lt"
    case "+" if is_string: return "."
```

Backend checks operand types to select Perl's string operators (`eq`, `ne`, `lt`, `gt`, `le`, `ge`, `.`) vs numeric operators (`==`, `!=`, `<`, `>`, `<=`, `>=`, `+`).

**Should be:** Lowering should emit `StringCompare` IR nodes for string comparisons and `StringConcat` for string concatenation, distinct from primitive `BinaryOp`.

---

### 4. Collection Containment Dispatch

**Location:** `perl.py:1696-1712` (`_containment_check`)

```python
if isinstance(container_type, Set):
    return f"{neg}exists({container_str}->{{{item_str}}})"
if isinstance(container_type, Map):
    return f"{neg}exists({container_str}->{{{item_str}}})"
if isinstance(container_type, Primitive) and container_type.kind == "string":
    return f"(index({container_str}, {item_str}) >= 0)"
# For arrays, use grep
return f"(grep {{ $_ eq {item_str} }} @{{{container_str}}})"
```

Backend switches on container type to emit correct containment semantics. Sets/maps use `exists()`, strings use `index()`, and arrays require `grep`.

**Should be:** Lowering should emit type-specific IR: `SetContains`, `MapContains`, `StringContains`, `SliceContains`.

---

### 5. Hoisted Variable Computation

**Location:** `perl.py:420-517` (`_collect_undeclared_assigns`, `_collect_undeclared_info`)

```python
def _collect_undeclared_assigns(self, stmts: list[Stmt]) -> set[str]:
    """Find variables that need pre-declaration at function scope.

    In Perl, declarations inside control flow blocks (if/while/for) are block-scoped,
    not visible to sibling blocks. We need to pre-declare vars that:
    1. Have is_declaration=False assignments anywhere
    2. Are assigned in TupleAssign where another target is hoisted
    3. Are declared inside control flow AND declared at top level
    """
```

The backend extensively traverses function bodies to compute which variables need pre-declaration. Perl's `my` inside blocks is block-scoped, unlike Python's function-scoped semantics.

**Should be:** Middleend hoisting pass should compute complete hoisting requirements per language. Backend should simply read `func.hoisted_vars` annotation.

---

### 6. Try-Catch Return Propagation

**Location:** `perl.py:519-549` (`_body_has_return`), `perl.py:1052-1119` (`_emit_try_catch`)

```python
has_return = self._body_has_return(body)
if has_return:
    self._line("my $_try_result;")
    self._line("my $_try_returned = 0;")
    # Wrap in labeled loop so 'last' can exit when returning
    self._line("TRYBLOCK: for (1) {")
    self._in_try_with_return = True
...
# Inside try body, transform return into flag pattern
if self._in_try_with_return:
    self._line(f"$_try_result = {self._expr(value)};")
    self._line("$_try_returned = 1;")
    self._line("last TRYBLOCK;")
```

Perl's `eval {}` block cannot use `return` to exit the enclosing function. Backend scans try bodies for returns and emits a flag-based workaround with labeled loops.

**Should be:** Lowering could emit `TryCatch` with `body_has_return: bool` annotation, or middleend should annotate try blocks. Backend shouldn't need to scan for return statements.

---

### 7. Struct Field Collection

**Location:** `perl.py:713` (`struct_fields`), `perl.py:1620-1638` (`StructLit` emission)

```python
self.struct_fields[struct.name] = [(f.name, f.typ) for f in struct.fields]
...
field_info = self.struct_fields.get(struct_name, [])
if field_info:
    ordered_args = []
    for field_name, field_type in field_info:
        if field_name in fields:
            ordered_args.append(self._expr(fields[field_name]))
```

Backend collects struct field info to emit constructor arguments in correct order for Perl's positional constructors.

**Should be:** `StructLit` IR node should already have fields in constructor parameter order, as determined by the fields phase.

---

### 8. Method Dispatch by Receiver Type

**Location:** `perl.py:1314-1447` (`MethodCall` handling)

The method implements a large dispatch table based on receiver type:
- String methods: join, split, upper, lower, find, rfind, startswith, endswith, replace
- Slice methods: append, extend, pop, copy
- Map methods: get

Each method has type-specific Perl implementations.

**Should be:** Lowering should emit type-specific method IR: `StringJoin`, `StringSplit`, `SliceAppend`, `SlicePop`, `MapGet`, etc.

---

### 9. String Iteration Special Handling

**Location:** `perl.py:963-1008` (`_emit_for_range`)

```python
if is_string:
    if idx is not None and val is not None:
        tmp = "_chars"
        self._line(f"my @{tmp} = split(//, {iter_expr});")
        self._line(f"for my ${idx} (0 .. $#{tmp}) {{")
        self._line(f"my ${val} = ${tmp}[${idx}];")
    elif val is not None:
        self._line(f"for my ${val} (split(//, {iter_expr})) {{")
```

Python strings are character sequences; Perl strings are scalar values. Backend uses `split(//, $str)` to iterate over characters.

**Should be:** Frontend should emit distinct IR nodes for character-based string operations (`CharAt`, `CharIter`). Currently ForRange must detect string type at emission time.

---

### 10. Bitwise Operations Detection

**Location:** `perl.py:562-652` (`_scan_for_bitwise`, `_body_has_bitwise`, `_stmt_has_bitwise`, `_expr_has_bitwise`)

```python
def _scan_for_bitwise(self, module: Module) -> None:
    """Scan module for bitwise operations to determine if `use integer;` is needed."""
    for func in module.functions:
        if self._body_has_bitwise(func.body):
            self._needs_integer = True
            return
```

Backend traverses the entire module (~90 lines of visitor code) to detect if any bitwise operations exist. Perl requires `use integer;` for correct integer bitwise semantics.

**Should be:** A middleend pass should annotate `Module.uses_bitwise_ops: bool`, or lowering should emit distinct `BitwiseOp` IR nodes that backends can detect during emission.

---

### 11. Negative Index Pattern Detection

**Location:** `perl.py:1714-1722` (`_negative_index`)

```python
def _negative_index(self, obj: Expr, index: Expr) -> str | None:
    """Detect len(obj) - N and return -N as string, or None if no match."""
    if not isinstance(index, BinaryOp) or index.op != "-":
        return None
    if not isinstance(index.left, Len) or not isinstance(index.right, IntLit):
        return None
    if self._expr(index.left.expr) != self._expr(obj):
        return None
    return f"-{index.right.value}"
```

Backend pattern-matches `len(obj) - N` expressions to emit Perl's native negative indexing (`$arr[-N]`).

**Should be:** Lowering should detect this pattern and emit `Index(obj, NegativeIndex(N))` or `LastElement(obj, offset=N)` IR nodes.

---

### 12. Function and Constant Name Tracking

**Location:** `perl.py:373-377`, `391-397`, `1168-1174`, `1303-1313`

```python
self._known_functions: set[str] = set()  # Module-level function names
self.constants: set[str] = set()
...
if name in self._known_functions:
    return f"\\&{_safe_name(name)}"
if name in self.constants:
    const_call = name.upper() + "()"
    if self.current_package is not None:
        return f"main::{const_call}"
```

Backend tracks known functions to emit correct reference syntax (`\&func` for function refs) and constants to emit with namespace qualifiers when inside packages.

**Should be:** IR `Var` nodes should have `is_function_ref: bool` and `is_constant: bool` annotations, or `FuncRef` should always be used for function references (not `Var`).

---

### 13. Function Parameter Callable Detection

**Location:** `perl.py:754-756`, `776`, `1306-1307`

```python
self._func_params = {p.name for p in func.params if isinstance(p.typ, FuncType)}
...
if func in self._func_params:
    return f"${safe_func}->({args_str})"
```

Backend tracks which parameters have `FuncType` to emit correct call syntax. Perl function references stored in scalars need `$ref->()` call syntax.

**Should be:** `Call` IR nodes should have a flag indicating when the callee is a function-typed variable vs a known function name, or lowering should emit `IndirectCall(var, args)` vs `Call(func_name, args)`.

---

### 14. Endswith Tuple Argument Expansion

**Location:** `perl.py:1668-1679` (`_endswith_expr`)

```python
def _endswith_expr(self, obj_str: str, args: list[Expr]) -> str:
    """Generate endswith check, handling tuple arguments."""
    if len(args) == 1 and isinstance(args[0], TupleLit):
        # Handle endswith with tuple: s.endswith((" ", "\n")) -> multiple checks
        checks = []
        for elem in args[0].elements:
            suffix = self._expr(elem)
            checks.append(f"(substr({obj_str}, -length({suffix})) eq {suffix})")
        return "(" + " || ".join(checks) + ")"
```

Python's `str.endswith()` accepts a tuple of suffixes. Backend expands this into multiple checks at emission time.

**Should be:** Lowering should detect tuple arguments to `endswith`/`startswith` and emit `StringEndsWithAny(string, suffixes)` IR nodes or expand to `BinaryOp("||", check1, check2, ...)`.

---

### 15. Regex Escaping in String Replace

**Location:** `perl.py:1376-1385`, `1985-2047` (`_escape_perl_regex`, `_escape_perl_replacement`, `_escape_regex_charclass`)

```python
if isinstance(args[0], StringLit):
    old_val = _escape_perl_regex(args[0].value)
else:
    old_val = self._expr(args[0])
...
return f"({obj_str} =~ s/{old_val}/{new_val}/gr)"
```

Backend implements ~60 lines of regex escaping logic to safely convert Python `str.replace()` to Perl's `s///` operator. Perl's replacement strings have different escaping rules than the pattern.

**Should be:** Lowering should emit `StringReplace(string, pattern, replacement, literal=True)` with explicit semantics. Backends that use regex-based replacement (Perl, Ruby) can apply appropriate escaping; others use literal replacement APIs.

---

## Python

The Python backend is the cleanest of all backends because the source language matches the target language. No semantic gaps need bridging—Python constructs map directly to Python constructs. The backend performs no compensations for frontend or middleend deficiencies.

### Uncompensated Deficiencies (Non-Idiomatic Output)

These are not compensations but areas where the output could be more idiomatic if frontend/middleend did additional work:

#### 1. `_intPtr` Helper Function

**Location:** `python.py:295-297`

```python
def _intPtr(val: int) -> int | None:
    return None if val == -1 else val
```

The backend emits a module-level helper function for sentinel-to-optional conversion instead of inlining the ternary expression at each use site.

**Should be:** Lowering could inline `None if val == -1 else val` at each call site, or frontend could emit `SentinelToOptional` nodes that backends can inline or emit as helpers based on target language idioms.

---

#### 2. Pointer/Optional Type Conflation

**Location:** `python.py:1032-1033`

```python
case Pointer(target=target):
    return self._type(target)
```

`Pointer(StructRef("X"))` renders as `X` but should be `X | None` when the field is nullable. Example: `word: Word = None` should be `word: Word | None = None`. Approximately 50 fields in typical codebases are affected.

**Should be:** Inference should distinguish `Pointer` (non-nullable reference) from `Optional(Pointer)` (nullable reference). Currently both use `Pointer` and backends must guess from context whether to emit nullable syntax.

---

### Not Compensations (Idiomatic Transforms)

The Python backend performs several pattern transformations that improve output quality but aren't compensating for deficiencies—they're optimizations that preserve Python idioms:

1. **Negative index folding** (`python.py:956-965`): Detects `len(obj) - N` and emits `-N`
2. **Tuple unpacking fold** (`python.py:578-612`): Converts `for _item in xs; a = _item[0]; b = _item[1]` to `for a, b in xs`
3. **Range pattern detection** (`python.py:1212-1254`): Converts `for i := 0; i < len(x); i++` to `for i in range(len(x))`

These are acceptable backend optimizations rather than compensations for missing frontend analysis.

---

## PHP

### 1. Bool-to-Int Casting

**Location:** `php.py:871-876` (floor div), `php.py:917-919` (bitwise NOT), `php.py:1057-1060` (abs), `php.py:1061-1082` (min/max), `php.py:1099-1105` (divmod), `php.py:1454-1457` (arithmetic ops)

```python
if op == "//" and (left.typ == BOOL or right.typ == BOOL):
    left_str = f"({self._expr(left)} ? 1 : 0)" if left.typ == BOOL else self._expr(left)
```

Multiple locations check if operands have `BOOL` type and insert ternary conversion (`? 1 : 0`) when bool values are used in:
- Floor division (`intdiv()`)
- Bitwise NOT (`~`)
- `abs()`, `min()`, `max()`, `divmod()`, `pow()` calls
- `MinExpr` and `MaxExpr` when mixed with non-bool

**Should be:** Inference or lowering should insert explicit `Cast(BOOL, INT)` nodes when bools flow into int-expecting operations.

---

### 2. Truthy Type Dispatch

**Location:** `php.py:853-865` (`Truthy` handling)

```python
if _is_string_type(inner_type):
    return f"({inner_str} !== '')"
if isinstance(inner_type, (Slice, Map, Set)):
    return f"(count({inner_str}) > 0)"
if isinstance(inner_type, Optional) and isinstance(inner_type.inner, (Slice, Map, Set)):
    return f"({inner_str} !== null && count({inner_str}) > 0)"
if isinstance(inner_type, Primitive) and inner_type.kind == "int":
    return f"({inner_str} !== 0)"
return f"({inner_str} !== null)"
```

The backend switches on the inner type of `Truthy` nodes to emit type-appropriate truthiness checks. PHP's truthiness semantics differ from Python's (e.g., `0` and `""` are falsy in PHP).

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `StringNonEmpty`, `SliceNonEmpty`, `IntNonZero`, `IsNotNil`, etc.

---

### 3. String Operator Detection

**Location:** `php.py:497-498` (OpAssign), `php.py:877-882` (BinaryOp +), `php.py:1490-1506` (`_binary_op`)

```python
if op == "+" and (_is_string_type(left.typ) or _is_string_type(right.typ)):
    return f"{left_str} . {right_str}"
```

The backend checks operand types to select PHP's string concatenation operator (`.`) vs numeric addition (`+`). Also changes `+=` to `.=` for string augmented assignment.

**Should be:** Lowering should emit `StringConcat(parts)` IR nodes for string concatenation, distinct from `BinaryOp("+")` for numeric addition.

---

### 4. Collection Containment Dispatch

**Location:** `php.py:1237-1248` (`_containment_check`)

```python
if isinstance(container_type, Set):
    return f"{neg}isset({container_str}[{item_str}])"
if isinstance(container_type, Map):
    return f"{neg}array_key_exists({item_str}, {container_str})"
if isinstance(container_type, Primitive) and container_type.kind == "string":
    return f"({neg}str_contains({container_str}, {item_str}))"
return f"{neg}in_array({item_str}, {container_str}, true)"
```

The backend switches on container type to emit correct containment semantics (sets use `isset()`, maps use `array_key_exists()`, strings use `str_contains()`, arrays use `in_array()`).

**Should be:** Lowering should emit type-specific IR: `SetContains`, `MapContains`, `StringContains`, `SliceContains`.

---

### 5. Struct Literal Field Ordering

**Location:** `php.py:246-260` (`_collect_struct_fields`), `php.py:1298-1317` (`_struct_lit`)

```python
field_info = self.sorted_struct_fields.get(struct_name, [])
if field_info:
    ordered_args = []
    for field_name, field_type in field_info:
        if field_name in fields:
            ordered_args.append(self._expr(fields[field_name]))
```

The backend collects struct field info (sorting required fields before optional) to emit constructor arguments in correct order for PHP's positional constructors.

**Should be:** `StructLit` IR node should already have fields in constructor parameter order, as determined by the fields phase.

---

### 6. Method Dispatch by Receiver Type

**Location:** `php.py:1118-1213` (`_method_call`)

The method implements a large dispatch table based on receiver type:
- Slice methods: `append`, `pop`, `copy`, `decode`, `extend`, `remove`, `clear`, `insert`
- String methods: `startswith`, `endswith`, `find`, `rfind`, `replace`, `split`, `join`, `lower`, `upper`
- Map methods: `get`

Each method has type-specific PHP implementations (e.g., `array_push()` vs `str_starts_with()`).

**Should be:** Lowering should emit type-specific method IR: `SliceAppend`, `SlicePop`, `StringStartsWith`, `StringJoin`, `MapGet`, etc.

---

### 7. Bool-Int Comparison Detection

**Location:** `php.py:884-897`

```python
# Use loose equality when comparing bool with int
if op in ("==", "!=") and _is_bool_int_compare(left, right):
    loose_op = "==" if op == "==" else "!="
    return f"{left_str} {loose_op} {right_str}"
# Use loose equality when comparing any-typed expr with bool
if op in ("==", "!=") and (isinstance(left.typ, InterfaceRef) and right.typ == BOOL) ...
```

PHP's strict equality (`===`) treats `true !== 1`. The backend detects bool/int comparisons and uses loose equality (`==`) to match Python semantics where `True == 1`.

**Should be:** Inference should annotate comparisons between bool and int with `use_loose_equality: bool`, or lowering should emit `LooseEquals` IR nodes when bool/int mixing is detected.

---

### 8. Ternary Nested Parentheses

**Location:** `php.py:927-935` (Ternary), `php.py:1440-1445` (`_cond_expr`)

```python
# Wrap logical ops in condition since || has lower precedence than ?:
cond_str = self._cond_expr(cond)
# PHP 8+ requires parens for nested ternaries
if isinstance(else_expr, Ternary):
    else_str = f"({self._expr(else_expr)})"
```

PHP 8+ requires parentheses for nested ternaries (associativity changed). The backend detects nested `Ternary` nodes and wraps them, also wrapping `||` conditions due to precedence issues.

**Should be:** Lowering could emit `Ternary` with an annotation for required parenthesization, or middleend could flatten chained ternaries into explicit `if-else` form.

---

### 9. String Find/Rfind Return Value Conversion

**Location:** `php.py:1152-1157`, `1197-1202`

```python
if method == "find":
    return f"(mb_strpos({obj_str}, {args_str}) === false ? -1 : mb_strpos({obj_str}, {args_str}))"
```

PHP's `mb_strpos()` returns `false` when not found; Python's `str.find()` returns `-1`. The backend emits a conditional to convert.

**Should be:** Lowering should emit `StringFind(string, needle, not_found_sentinel=-1)` with explicit return-value semantics that backends can implement appropriately.

---

### 10. UnaryOp Negation Special Cases

**Location:** `php.py:902-926`

```python
case UnaryOp(op="!", operand=operand):
    operand_type = operand.typ
    if isinstance(operand_type, Primitive) and operand_type.kind == "int":
        return f"({self._expr(operand)} === 0)"
    if isinstance(operand_type, (InterfaceRef, StructRef, Pointer)):
        return f"({self._expr(operand)} === null)"
```

The backend checks operand types for `!` to emit type-appropriate falsiness checks: `=== 0` for int, `=== null` for objects (PHP's `!` operator has different semantics than Python's `not`).

**Should be:** Lowering should emit `Falsy(expr)` or specialize `UnaryOp("!")` into type-specific IR based on operand type.

---

### 11. String Iteration Special Handling

**Location:** `php.py:651-709` (`_emit_for_range`)

```python
is_string = isinstance(iter_type, Primitive) and iter_type.kind == "string"
if is_string:
    if value is not None and index is not None:
        self._line(f"for (${idx} = 0; ${idx} < mb_strlen({iter_expr}); ${idx}++)")
        self._line(f"${val} = mb_substr({iter_expr}, ${idx}, 1);")
```

Python strings are character sequences; PHP strings are byte sequences. The backend uses `mb_strlen`/`mb_substr`/`mb_str_split` for all string iteration to handle Unicode correctly.

**Should be:** Frontend should emit distinct IR nodes for character-based string iteration. `ForRange` could carry `is_string_iter: bool` annotation, or lowering should emit `ForStringChars(string, index, value, body)`.

---

### 12. Pop Return Value at Index

**Location:** `php.py:1124-1130`

```python
if method == "pop":
    if not args:
        return f"array_pop({obj_str})"
    idx = self._expr(args[0])
    if idx == "0":
        return f"array_shift({obj_str})"
    return f"array_splice({obj_str}, {idx}, 1)[0]"
```

Python's `list.pop(i)` removes and returns element at index `i`. PHP's `array_pop()` only pops from end. The backend pattern-matches index arguments to emit `array_shift()` for index 0 or `array_splice(...)[0]` for arbitrary indices.

**Should be:** Lowering should emit `ListPop(list, index)` IR node with explicit semantics, rather than relying on backends to pattern-match `MethodCall("pop")`.

---

### 13. Bytes/String Conversion

**Location:** `php.py:1134-1136` (bytes.decode), `php.py:1253-1260` (bytes cast), `php.py:1274-1281` (str from bytes)

```python
if method == "decode":
    return f"UConverter::transcode(pack('C*', ...{obj_str}), 'UTF-8', 'UTF-8')"
```

The backend implements Python's `bytes.decode()` and `str(bytes)` using `UConverter::transcode()` for proper UTF-8 handling with replacement characters.

**Should be:** Lowering should emit `BytesDecode(bytes_expr, encoding, errors)` IR nodes for bytes-to-string conversion.

---

### 14. Endswith Tuple Argument Expansion

**Location:** `php.py:1146-1151`, `1189-1194`

```python
if method == "endswith":
    if args and isinstance(args[0], TupleLit):
        checks = " || ".join(
            f"str_ends_with({obj_str}, {self._expr(e)})" for e in args[0].elements
        )
        return f"({checks})"
```

Python's `str.endswith()` accepts a tuple of suffixes. The backend expands this into multiple `str_ends_with()` checks joined with `||`.

**Should be:** Lowering should detect tuple arguments to `endswith`/`startswith` and emit `StringEndsWithAny(string, suffixes)` IR nodes or expand to `BinaryOp("||", check1, check2, ...)`.

---

## Ruby

### 1. Bool-to-Int Casting

**Location:** `ruby.py:950-951` (abs), `964-970` (min), `980-985` (max), `1018-1028` (divmod), `1031-1061` (pow), `1543-1557` (floor div), `1562-1605` (bool comparisons), `1630-1640` (exponent), `1656-1671` (ordered comparisons), `1672-1685` (shifts), `1716-1736` (arithmetic), `1799-1824` (MinExpr/MaxExpr), `1825-1832` (UnaryOp)

```python
if left.typ == BOOL:
    left_str = self._coerce_bool_to_int(left, raw=True)
# produces: "(expr ? 1 : 0)"
```

Multiple locations check if operands have `BOOL` type and insert ternary conversion (`? 1 : 0`) when bool values are used in:
- Arithmetic operators (+, -, *, /, %, //)
- Comparison operators (<, >, <=, >=, ==, !=)
- Bitwise operators (&, |, ^)
- Shift operators (<<, >>)
- Exponentiation (**)
- Unary operators (-, ~)
- Built-in functions (min, max, abs, pow, divmod)
- MinExpr/MaxExpr nodes

**Should be:** Inference or lowering should insert explicit `Cast(BOOL, INT)` nodes when bools flow into int-expecting operations.

---

### 2. Truthy Type Dispatch

**Location:** `ruby.py:1476-1525` (`Truthy` handling)

```python
if isinstance(inner_type, Slice) or isinstance(inner_type, Map) or isinstance(inner_type, Set):
    return f"({expr_str} && !{expr_str}.empty?)"
if inner_type == INT:
    return f"{expr_str} != 0"
if inner_type == FLOAT:
    return f"{expr_str} != 0.0"
```

The backend switches on the inner type of `Truthy` nodes to emit type-appropriate truthiness checks. Ruby's truthiness semantics (only `nil` and `false` are falsy) differ significantly from Python's.

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `StringNonEmpty`, `SliceNonEmpty`, `IntNonZero`, `FloatNonZero`, `IsNotNil`, etc.

---

### 3. Map Key Type Coercion (Python Key Equivalence)

**Location:** `ruby.py:2197-2240` (`_coerce_map_key`), `ruby.py:1-7` (acknowledged limitation in docstring)

```python
# BOOL variable → INT
if map_key == "int" and key_typ == "bool":
    return f"({key_code} ? 1 : 0)"
# FLOAT variable → INT
if map_key == "int" and key_typ == "float":
    return f"({key_code}).to_i"
```

Python treats `True==1` and `False==0` as equivalent dict keys. Ruby `Hash` treats them as different. The backend coerces keys at runtime based on declared map key type. The docstring explicitly notes this only works in VarDecl initializers and direct Index/method access—not in Assign, function args, return statements, or comprehensions.

**Should be:** Inference should track Python's key equivalence semantics, and lowering should emit explicit `CoerceMapKey(expr, target_key_type)` nodes when key types don't match the map's declared key type.

---

### 4. Collection Containment Dispatch

**Location:** `ruby.py:1743-1746` (`in` operator in BinaryOp)

```python
if op == "in":
    return f"{self._expr(right)}.include?({self._expr(left)})"
```

The backend uses Ruby's `.include?` method for all containment checks. While Ruby's method works across types, the emission is uniform rather than type-specialized.

**Should be:** Lowering should emit type-specific IR: `SetContains`, `MapContains`, `SliceContains`, `StringContains`.

---

### 5. String Split Semantics

**Location:** `ruby.py:1377-1404` (`split` and `rsplit` method handling)

```python
if len(args) == 1:
    # Ruby returns [] for "".split(x), Python returns [""]
    return f'({obj_str}.empty? ? [""] : {obj_str}.split(Regexp.new(Regexp.escape({arg_str})), -1))'
else:
    # Python maxsplit=0 means no splits, Ruby limit=1 means no splits
    # Python maxsplit=n means n splits (n+1 parts), Ruby limit=n+1
    return f'({obj_str}.empty? ? [""] : {obj_str}.split(Regexp.new(Regexp.escape({sep_str})), {maxsplit_str} == 0 ? 1 : {maxsplit_str} + 1))'
```

The backend emits complex patterns to match Python's `split()` behavior:
- Empty string handling (`"".split(x)` returns `[""]` in Python, `[]` in Ruby)
- Maxsplit argument semantics (Python's maxsplit is splits count, Ruby's limit is parts count)
- Regex escaping for literal pattern matching

**Should be:** Lowering should emit `StringSplit(string, sep, maxsplit)` with explicit Python semantics, or a `PythonSplit` IR node distinct from `SimpleSplit`.

---

### 6. Method Dispatch by Receiver Type

**Location:** `ruby.py:1142-1459` (`MethodCall` handling)

The method implements a large dispatch table based on receiver type:
- String methods: join, split, rsplit, startswith, endswith, find, rfind, replace, count, isupper, islower, title, zfill, splitlines, expandtabs, casefold, format, removeprefix, removesuffix
- Slice methods: pop, extend, index, remove, insert, sort
- Map methods: get, items, keys, values, copy, setdefault, popitem, pop
- Set methods: remove, discard, pop, copy, issubset, issuperset, isdisjoint, update, symmetric_difference, union, intersection, difference

Each method has type-specific Ruby implementations.

**Should be:** Lowering should emit type-specific method IR: `StringJoin`, `StringSplit`, `SliceAppend`, `SlicePop`, `MapGet`, `SetAdd`, etc.

---

### 7. Negative Index Pattern Detection

**Location:** `ruby.py:2170-2178` (`_negative_index`)

```python
def _negative_index(self, obj: Expr, index: Expr) -> str | None:
    """Detect len(obj) - N and return -N as string, or None if no match."""
    if not isinstance(index, BinaryOp) or index.op != "-":
        return None
    if not isinstance(index.left, Len) or not isinstance(index.right, IntLit):
        return None
    if self._expr(index.left.expr) != self._expr(obj):
        return None
    return f"-{index.right.value}"
```

The backend pattern-matches `len(obj) - N` expressions to emit Ruby's native negative indexing (`arr[-N]`).

**Should be:** Lowering should detect this pattern and emit `Index(obj, NegativeIndex(N))` or `LastElement(obj, offset=N)` IR nodes.

---

### 8. And/Or Value Preservation

**Location:** `ruby.py:1526-1541` (value-returning and/or), `2390-2454` (helper methods)

```python
case BinaryOp(op="&&", left=left_expr, right=right_expr) if self._is_value_and_or(left_expr):
    # Python's `and` returns first falsy value or last value
    # Ruby: python_truthy(x) ? y : x
    left_val, left_str, cond = self._extract_and_or_value(left_expr)
    right_str = self._and_or_operand(right_expr)
    return f"({cond} ? {right_str} : {left_str})"
```

Python's `and`/`or` operators return actual values, not just booleans. The backend implements complex logic to preserve these semantics, including truthy checks for different types and chained operations.

**Should be:** Lowering should emit distinct `PythonAnd(left, right)` and `PythonOr(left, right)` IR nodes that explicitly return values, separate from `BinaryOp("&&")` which returns boolean.

---

### 9. Array Comparison Operators

**Location:** `ruby.py:1641-1655`

```python
case BinaryOp(op=op, left=left, right=right) if op in ("<", ">", "<=", ">=") and isinstance(left.typ, (Slice, Tuple)):
    # Ruby arrays don't have <, >, <=, >= - use <=> spaceship
    left_str = self._expr(left)
    right_str = self._expr(right)
    return f"(({left_str} <=> {right_str}) {op} 0)"
```

Ruby arrays don't support `<`, `>`, `<=`, `>=` operators directly. The backend uses the spaceship operator `<=>` and compares the result to 0.

**Should be:** Lowering should emit `ArrayCompare(left, right, op)` IR nodes for array ordering comparisons, distinct from primitive `BinaryOp`.

---

### 10. Slice Bounds Clamping

**Location:** `ruby.py:2077-2168` (`_slice_expr`)

```python
# Clamp negative start indices: Ruby returns nil if negative index goes past start
# Python clamps to 0, so items[-100:2] gives items[0:2]
if clamp_low:
    low_str = f"[{low_str}, -{obj_str}.length].max"
# Ruby returns nil for out-of-bounds slices, Python returns []
# Use || [] to match Python semantics
return f"({obj_str}[{low_str}...{high_str}] || [])"
```

The backend implements extensive logic to match Python's slice semantics:
- Clamping negative indices that exceed sequence length
- Returning empty array instead of nil for out-of-bounds slices
- Handling step parameters with reverse iteration
- Different semantics for bytes vs arrays vs strings

**Should be:** Lowering should emit `Substring` or `SliceExpr` with explicit semantics flag (`clamp=True`, `empty_on_oob=True`), or emit the bounds-checking wrapper at lowering time.

---

### 11. String/Array Multiplication with Negative

**Location:** `ruby.py:1686-1715`

```python
case BinaryOp(op="*", left=left, right=right) if left.typ == Primitive(kind="string") and right.typ == INT:
    # Handle negative multiplier: [n, 0].max
    left_str = self._expr(left)
    right_str = self._expr(right)
    return f"{left_str} * [{right_str}, 0].max"
```

Python returns empty string/array for negative multipliers; Ruby raises an error. The backend clamps the multiplier to 0. Also handles `int * string` order reversal (Ruby requires string on left).

**Should be:** Lowering should emit `StringRepeat(string, count)` or `ArrayRepeat(array, count)` IR nodes with explicit clamping semantics.

---

### 12. Dict View Set Operations

**Location:** `ruby.py:1782-1789`, `2533-2539`

```python
# Detect set operations on dict views (keys/items/values)
# These return arrays in Ruby but sets in Python
if op in ("&", "|", "-", "^") and (_is_dict_view(left) or _is_dict_view(right)):
    self._needs_set = True
    left_set = f"Set[*{left_str}]" if _is_dict_view(left) else left_str
    right_set = f"Set[*{right_str}]" if _is_dict_view(right) else right_str
    return f"{left_set} {rb_op} {right_set}"
```

Python dict views (`.keys()`, `.values()`, `.items()`) support set operations. The backend detects view expressions and wraps them in `Set[]` for set operators.

**Should be:** Lowering should convert dict view expressions to explicit set types when used in set operations, or emit `DictKeysView`, `DictValuesView`, `DictItemsView` IR nodes with set-compatible semantics.

---

### 13. Find/Rfind Return Value Conversion

**Location:** `ruby.py:1365-1369` (rfind), `ruby.py:1747-1773` (find == -1 pattern)

```python
# Python: s.rfind(x) returns -1 if not found, Ruby rindex returns nil
if method == "rfind" and receiver_type == STRING:
    return f"({obj_str}.rindex({arg_str}) || -1)"
# Python: s.find(x) == -1 -> Ruby: s.index(x).nil?
if op in ("==", "!="):
    if find_expr is not None:
        if op == "==":
            return f"{obj_str}.index({args_str}).nil?"
```

The backend converts between Ruby's `nil` and Python's `-1` for find operations, and optimizes the common `find(x) == -1` pattern to use `.nil?`.

**Should be:** Lowering should emit `StringFind(string, needle, not_found_sentinel=-1)` with explicit return-value semantics.

---

### 14. List Insert Index Handling

**Location:** `ruby.py:1341-1351`

```python
# Ruby's insert(100, x) on a 5-element array pads with nil, Python clips
# Ruby's insert(-1, x) appends, Python inserts before last
return f"(-> {{ _idx = {idx_str}; {obj_str}.insert([[_idx < 0 ? {obj_str}.length + _idx : _idx, 0].max, {obj_str}.length].min, {val_str}) }}).call"
```

The backend emits complex index clamping logic for `list.insert()` because Ruby and Python have different out-of-bounds behavior.

**Should be:** Lowering should emit `ListInsert(list, index, value, clamp=True)` with explicit clamping semantics.

---

### 15. Range Helper for Negative Step

**Location:** `ruby.py:354-357`, `1120-1124`

```python
helper = "def _range(start, stop = nil, step = 1); stop.nil? ? (0...start).step(step).to_a : (step > 0 ? (start...stop).step(step).to_a : (stop + 1..start).step(-step).to_a.reverse); end"
...
return f"_range({args_str})"
```

The backend emits a helper function to handle `range(start, stop, step)` with negative steps. Ruby ranges don't natively support negative stepping.

**Should be:** Lowering should emit `Range(start, stop, step)` with a flag indicating negative step handling, or expand negative-step ranges into explicit iteration patterns during lowering.

---

### 16. Endswith/Startswith Tuple Argument Expansion

**Location:** `ruby.py:1183-1189`

```python
if method in ("startswith", "endswith") and len(args) == 1:
    if isinstance(args[0], TupleLit):
        obj_str = self._expr(obj)
        unpacked = ", ".join(self._expr(e) for e in args[0].elements)
        rb_method = _method_name(method, receiver_type)
        return f"{obj_str}.{rb_method}({unpacked})"
```

Python's `str.endswith()` accepts a tuple of suffixes. Ruby's `end_with?` accepts multiple arguments. The backend unpacks the tuple for multi-argument call.

**Should be:** Lowering should detect tuple arguments to `endswith`/`startswith` and emit `StringEndsWithAny(string, suffixes)` IR nodes or unpack to multi-argument form during lowering.

---

## Rust

### 1. Bool-to-Int Casting

**Location:** `rust.py:201-216` (`_is_bool`, `_needs_bool_int_coerce`), `rust.py:627-642` (MinExpr/MaxExpr), `rust.py:814-842` (BinaryOp arithmetic/comparison/bitwise), `rust.py:862-884` (UnaryOp), `rust.py:925` (int from bool), `rust.py:934-951` (abs/pow with bool), `rust.py:957-961` (divmod), `rust.py:1246-1249` (`_coerce_bool_to_int`)

```python
if _needs_bool_int_coerce(expr.left, expr.right):
    left = f"({self._emit_expr(expr.left)} as i64)"
```

Multiple locations check if operands have `BOOL` type and insert `as i64` casts when bool values are used in:
- Arithmetic operators (+, -, *, /, %)
- Comparison operators (<, >, <=, >=, ==, !=) when mixed with int
- Bitwise operators (&, |, ^) when mixed with int
- Shift operators (<<, >>)
- Unary operators (-, ~)
- Built-in functions (min, max, abs, pow, divmod)
- MinExpr/MaxExpr nodes

**Should be:** Inference or lowering should insert explicit `Cast(BOOL, INT)` nodes when bools flow into int-expecting operations.

---

### 2. Truthy Type Dispatch

**Location:** `rust.py:1251-1271` (`_emit_Truthy`)

```python
if isinstance(inner_type, Map):
    return f"!{inner}.is_empty()"
if isinstance(inner_type, Slice):
    return f"!{inner}.is_empty()"
if inner_type == STRING:
    return f"!{inner}.is_empty()"
if inner_type == INT:
    return f"({inner} != 0)"
```

The backend switches on the inner type of `Truthy` nodes to emit type-appropriate truthiness checks (`.is_empty()` for collections/strings, `!= 0` for int, direct value for bool).

**Should be:** Lowering should specialize `Truthy` into type-specific IR: `StringNonEmpty`, `SliceNonEmpty`, `MapNonEmpty`, `IntNonZero`, etc.

---

### 3. Collection Containment Dispatch

**Location:** `rust.py:714-740` (BinaryOp "in" and "not in")

```python
if isinstance(right_type, Map):
    return f"{right}.contains_key(&{left})"
if isinstance(right_type, Set):
    return f"{right}.contains(&{left})"
if isinstance(right_type, Slice):
    return f"{right}.contains(&{left})"
if right_type == STRING:
    return f"{right}.contains({left})"
```

The backend switches on container type to emit correct containment method (`.contains_key(&)` for Map, `.contains(&)` for Set/Slice, `.contains()` for String).

**Should be:** Lowering should emit type-specific IR: `SetContains`, `MapContains`, `SliceContains`, `StringContains`.

---

### 4. String Concatenation Detection

**Location:** `rust.py:810-811`, `847-860` (`_emit_string_add`, `_flatten_string_add`)

```python
if op == "+" and expr.typ == STRING:
    return self._emit_string_add(expr)
...
def _emit_string_add(self, expr: BinaryOp) -> str:
    parts: list[Expr] = []
    self._flatten_string_add(expr, parts)
    return f'format!("{placeholders}", {args})'
```

Rust's `&str + &str` doesn't work directly. The backend checks if operands are strings and emits `format!()` macro calls, flattening chained string concatenation into a single call.

**Should be:** Lowering should emit `StringConcat(parts)` IR nodes for string concatenation, distinct from `BinaryOp("+")` for numeric addition.

---

### 5. Empty Collection Type Annotation

**Location:** `rust.py:391-394` (VarDecl empty MapLit), `rust.py:404-407` (VarDecl Slice/Map/Set), `rust.py:1043-1069` (MapLit emission), `rust.py:745-763` (comparison with empty MapLit)

```python
# Empty map in VarDecl needs explicit type
if isinstance(s.value, MapLit) and isinstance(s.typ, Map) and not s.value.entries:
    typ = self._type_to_rust(s.typ)
    self.line(f"let {mut}{name}: {typ} = std::collections::HashMap::new();")
...
# Comparison with empty map needs turbofish syntax
return f"{left} {op} std::collections::HashMap::<{key_type}, {val_type}>::new()"
```

Rust requires explicit type annotations for empty collections. The backend inspects `VarDecl.typ` and emits turbofish annotations (`HashMap::<K, V>::new()`), and handles empty MapLit in comparisons specially.

**Should be:** Lowering should emit typed empty collection literals with explicit element types, or empty collections should carry their inferred type annotation from inference.

---

### 6. Dict View Set Operations

**Location:** `rust.py:690-712`

```python
if left_is_keys and right_is_keys:
    left_set = f"{left}.into_iter().collect::<std::collections::HashSet<_>>()"
    right_set = f"{right}.into_iter().collect::<std::collections::HashSet<_>>()"
    if op == "&":
        return f"{left_set}.intersection(&{right_set}).cloned().collect::<std::collections::HashSet<_>>()"
```

The backend detects set operations (`&`, `|`, `^`, `-`) on dict `.keys()` or `.items()` views and converts them to HashSet operations.

**Should be:** Lowering should convert dict view expressions to explicit set types when used in set operations, or emit `DictKeysView`, `DictItemsView` IR nodes with set-compatible semantics.

---

### 7. Map Optional Value Handling

**Location:** `rust.py:396-402` (VarDecl with Optional values), `rust.py:1071-1083` (`_emit_MapLit_with_optional`), `rust.py:1176-1183` (get with default on Optional-valued map)

```python
if isinstance(s.typ, Map) and isinstance(s.typ.value, Optional):
    val = self._emit_MapLit_with_optional(s.value, s.typ.value)
...
def _emit_MapLit_with_optional(self, expr: MapLit, opt_type: Optional) -> str:
    # Wrap non-None values in Some()
    if isinstance(v, NilLit):
        val = "None"
    else:
        val = f"Some({self._emit_expr(v)})"
```

When a map has `Optional` value type, the backend wraps non-None values in `Some()` during MapLit emission, and adjusts `get()` with default to also wrap in `Some()`.

**Should be:** Lowering should emit explicit `Some(value)` wrappers when assigning to Optional-typed map values, rather than relying on backends to detect this pattern.

---

### 8. Optional Comparison Wrapping

**Location:** `rust.py:765-808` (BinaryOp == and != with map.get())

```python
# Check if either side is a MethodCall returning Option (get without default)
if left_returns_option and not right_returns_option:
    return f"{left} {op} Some({right})"
if right_returns_option and not left_returns_option:
    return f"Some({left}) {op} {right}"
```

The backend detects comparisons where one side is a `map.get()` result (which returns `Option<V>` in Rust) and wraps the other side in `Some()` to make types compatible.

**Should be:** Inference should track Option-typed expressions, and lowering should emit explicit `Some(expr)` wrappers or `OptionEquals(opt_expr, value)` IR nodes.

---

### 9. Method Dispatch by Receiver Type

**Location:** `rust.py:1124-1225` (`_emit_MethodCall`)

The method implements a large dispatch table based on receiver type:
- String methods: upper, lower, strip, lstrip, rstrip, startswith, endswith, find, replace, split, join
- Map methods: get, keys, values, items, pop, setdefault, update, clear, copy, popitem
- Slice methods: append (→ push)

Each method has type-specific Rust implementations.

**Should be:** Lowering should emit type-specific method IR: `StringUpper`, `StringSplit`, `SliceAppend`, `MapGet`, `MapPop`, etc.

---

### 10. IsNil for Map.get() Results

**Location:** `rust.py:1272-1286` (`_emit_IsNil`)

```python
if isinstance(expr.expr, MethodCall) and expr.expr.method == "get":
    obj = self._emit_expr(expr.expr.obj)
    key = self._emit_expr(expr.expr.args[0])
    if expr.negated:
        return f"{obj}.get(&{key}).is_some()"
    return f"{obj}.get(&{key}).is_none()"
```

The backend special-cases `IsNil` when the inner expression is a `map.get()` call, emitting `.is_none()`/`.is_some()` on the raw get result rather than the emitted expression (which would have `.copied()` appended).

**Should be:** Lowering should distinguish `IsNil` (null check) from `IsEmpty` and emit appropriate IR. `map.get()` should have explicit Optional-returning semantics in IR.

---

### 11. Dict Merge Operator

**Location:** `rust.py:441-443` (OpAssign |=), `rust.py:679-688` (BinaryOp |)

```python
# Dict merge operator |= uses extend
if s.op == "|" and isinstance(s.value.typ, Map):
    self.line(f"{target}.extend({val}.iter().map(|(k, v)| (k.clone(), *v)));")
...
# Dict merge operator |
if op == "|" and isinstance(expr.left.typ, Map):
    return f"{{ let mut m = {left}.clone(); m.extend({right}.iter().map(|(k, v)| (k.clone(), *v))); m }}"
```

The backend detects the `|` and `|=` operators on Map types and emits Rust-specific merge patterns using `.extend()` and cloning.

**Should be:** Lowering should emit `MapMerge(left, right)` and `MapMergeInPlace(target, value)` IR nodes for dict merge operations.

---

### 12. Dict.fromkeys() Static Call

**Location:** `rust.py:1126-1132` (MethodCall on dict), `rust.py:1336-1350` (`_emit_StaticCall`)

```python
if isinstance(expr.obj, Var) and expr.obj.name == "dict" and expr.method == "fromkeys":
    return f"{keys}.iter().map(|k| (k.clone(), {value}.clone())).collect::<std::collections::HashMap<_, _>>()"
...
if isinstance(expr.on_type, Map) and method == "fromkeys":
    return f"{keys}.iter().map(|k| (k.clone(), {value})).collect::<std::collections::HashMap<_, _>>()"
```

The backend special-cases `dict.fromkeys()` as both a MethodCall on a `dict` Var and a StaticCall on Map type, emitting iterator-based construction.

**Should be:** Lowering should emit `MapFromKeys(keys, default_value)` IR node for this pattern.

---

### 13. Mutable Dict Value Access

**Location:** `rust.py:1135-1140` (MethodCall append on dict value)

```python
if isinstance(expr.obj, Index) and isinstance(expr.obj.obj.typ, Map):
    if isinstance(expr.receiver_type, Slice) and expr.method == "append":
        return f"{dict_var}.get_mut(&{key}).unwrap().push({val})"
```

When calling mutating methods (like `append`) on a dict value (`d[key].append(x)`), the backend detects this pattern and uses `.get_mut()` instead of `.get()` to obtain a mutable reference.

**Should be:** Lowering should emit `MutableDictAccess(dict, key)` when the result will be mutated, or annotate `Index` nodes with `mutable_access: bool`.

---

### 14. Nested Dict Index Access

**Location:** `rust.py:569-576` (IndexLV for nested maps), `rust.py:1095-1102` (Index for nested maps)

```python
# Nested map indexing - need get_mut for mutable access to inner map
if isinstance(lv.obj, Index) and isinstance(lv.obj.obj.typ, Map):
    inner_map = lv.obj.obj.typ
    if isinstance(inner_map.value, Map):
        return f"*{outer_obj}.get_mut(&{outer_key}).unwrap().entry({idx}).or_insert(Default::default())"
```

The backend detects nested dict access (`d[k1][k2]`) and emits appropriate `.get_mut()` chains for Rust's ownership system.

**Should be:** Lowering should emit `NestedMapIndex(outer_map, outer_key, inner_key)` or annotate `Index` nodes with nesting depth information.

---

### 15. Pop Return Value at Index

**Location:** `rust.py:1191-1196` (map.pop)

```python
if method == "pop":
    key = self._emit_expr(expr.args[0])
    if len(expr.args) == 1:
        return f"{obj}.remove(&{key}).unwrap()"
    else:
        default = self._emit_expr(expr.args[1])
        return f"{obj}.remove(&{key}).unwrap_or({default})"
```

Python's `dict.pop(key)` removes and returns the value. Rust's HashMap has `.remove()` which returns `Option<V>`. The backend emits `.unwrap()` or `.unwrap_or(default)` based on argument count.

**Should be:** Lowering should emit `MapPop(map, key, default?)` IR node with explicit return-value semantics.
