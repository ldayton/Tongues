"""Shared utilities for backend code emitters."""

from __future__ import annotations


# Go reserved words that need renaming
GO_RESERVED = frozenset(
    {
        "break",
        "case",
        "chan",
        "const",
        "continue",
        "default",
        "defer",
        "else",
        "fallthrough",
        "for",
        "func",
        "go",
        "goto",
        "if",
        "import",
        "interface",
        "map",
        "package",
        "range",
        "return",
        "select",
        "struct",
        "switch",
        "type",
        "var",
    }
)


def _upper_first(s: str) -> str:
    """Uppercase the first character of a string."""
    return (s[0].upper() + s[1:]) if s else ""


def go_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase for Go. Private methods (underscore prefix) become unexported."""
    is_private = name.startswith("_")
    if is_private:
        name = name[1:]
    parts = name.split("_")
    # Use upper on first char only (not capitalize which lowercases rest)
    result = "".join(_upper_first(p) for p in parts)
    # All-caps names (constants) stay all-caps even if originally private
    if name.isupper():
        return result
    if is_private:
        # Make first letter lowercase for unexported (private) names
        return result[0].lower() + result[1:] if result else result
    return result


def go_to_camel(name: str) -> str:
    """Convert snake_case to camelCase for Go."""
    if name == "self":
        return name
    if name.startswith("_"):
        name = name[1:]
    parts = name.split("_")
    if not parts:
        return name
    # All-caps names (constants) should use PascalCase in Go
    if name.isupper():
        return "".join(_upper_first(p) for p in parts)
    result = parts[0] + "".join(_upper_first(p) for p in parts[1:])
    # Handle Go reserved words
    if result in GO_RESERVED:
        return result + "_"
    return result


def to_snake(name: str) -> str:
    """Convert camelCase/PascalCase to snake_case."""
    if name.startswith("_"):
        name = name[1:]
    if "_" in name or name.islower():
        return name.lower()
    result: list[str] = []
    i = 0
    while i < len(name):
        ch = name[i]
        if ch.isupper() and i > 0:
            prev = name[i - 1]
            if prev.islower() or prev.isdigit():
                result.append("_")
            elif prev.isupper() and i + 1 < len(name) and name[i + 1].islower():
                result.append("_")
        result.append(ch)
        i += 1
    return "".join(result).lower()


def replace_format_placeholders(template: str, replacement: str) -> str:
    """Replace {0}, {1}, ... placeholders with a fixed replacement string."""
    result: list[str] = []
    i = 0
    while i < len(template):
        if template[i] == "{" and i + 1 < len(template) and template[i + 1].isdigit():
            j = i + 1
            while j < len(template) and template[j].isdigit():
                j += 1
            if j < len(template) and template[j] == "}":
                result.append(replacement)
                i = j + 1
                continue
        result.append(template[i])
        i += 1
    return "".join(result)


def to_camel(name: str) -> str:
    """Convert snake_case to camelCase, preserving leading underscores."""
    prefix = ""
    if name.startswith("_"):
        prefix = "_"
        name = name[1:]
    if "_" not in name:
        return prefix + (name[0].lower() + name[1:] if name else name)
    parts = name.split("_")
    return prefix + parts[0].lower() + "".join(p.capitalize() for p in parts[1:])


def to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase."""
    if name.startswith("_"):
        name = name[1:]
    parts = name.split("_")
    return "".join(p.capitalize() for p in parts)


def to_screaming_snake(name: str) -> str:
    """Convert to SCREAMING_SNAKE_CASE."""
    return to_snake(name).upper()


def escape_string(value: str) -> str:
    """Escape a string for use in a string literal (without quotes)."""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\f", "\\f")
        .replace("\v", "\\v")
        .replace("\x00", "\\x00")
        .replace("\x01", "\\u0001")
        .replace("\x7f", "\\u007f")
    )


class Emitter:
    """Base class for code emitters with indentation tracking."""

    def __init__(self, indent_str: str = "    ") -> None:
        self.indent: int = 0
        self.lines: list[str] = []
        self._indent_str = indent_str

    def line(self, text: str = "") -> None:
        """Emit a line with current indentation."""
        if text:
            self.lines.append(self._indent_str * self.indent + text)
        else:
            self.lines.append("")

    def output(self) -> str:
        """Return the accumulated output as a string."""
        return "\n".join(self.lines)
