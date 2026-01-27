"""Tongues CLI."""

import argparse
import sys
from pathlib import Path

from tongues.verify import verify_file


def find_python_files(path: Path) -> list[Path]:
    """Find all Python files in a directory."""
    if path.is_file():
        return [path] if path.suffix == ".py" else []
    return sorted(path.rglob("*.py"))


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify Python files against the Tongues subset."""
    paths: list[Path] = []
    for p in args.paths:
        paths.extend(find_python_files(Path(p)))

    if not paths:
        print("No Python files found", file=sys.stderr)
        return 1

    errors: list[tuple[Path, int, int, str, str]] = []
    warnings: list[tuple[Path, int, int, str, str]] = []
    all_seen_nodes: dict[str, int] = {}
    for path in paths:
        try:
            result = verify_file(path)
            for v in result.violations:
                item = (v.file, v.line, v.col, v.category, v.message)
                if v.is_warning:
                    warnings.append(item)
                else:
                    errors.append(item)
            for node_type, count in result.seen_nodes.items():
                all_seen_nodes[node_type] = all_seen_nodes.get(node_type, 0) + count
        except SyntaxError as e:
            print(f"Syntax error in {path}: {e}", file=sys.stderr)
            return 1

    if args.verbose:
        print(f"AST nodes seen:")
        for node_type in sorted(all_seen_nodes.keys()):
            print(f"  {node_type}: {all_seen_nodes[node_type]}")
        print()

    if not errors and not warnings:
        if args.verbose:
            print(f"Verified {len(paths)} file(s): OK")
        return 0

    # Group by category if requested
    if args.by_category:
        if errors:
            by_cat: dict[str, list[tuple[Path, int, int, str]]] = {}
            for file, line, col, cat, msg in errors:
                by_cat.setdefault(cat, []).append((file, line, col, msg))
            for cat in sorted(by_cat.keys()):
                print(f"\n[{cat}] ({len(by_cat[cat])} errors)")
                for file, line, col, msg in sorted(by_cat[cat]):
                    print(f"  {file}:{line}:{col}: {msg}")
        if warnings:
            by_cat = {}
            for file, line, col, cat, msg in warnings:
                by_cat.setdefault(cat, []).append((file, line, col, msg))
            for cat in sorted(by_cat.keys()):
                print(f"\n[{cat}] ({len(by_cat[cat])} warnings)")
                for file, line, col, msg in sorted(by_cat[cat]):
                    print(f"  {file}:{line}:{col}: warning: {msg}")
    else:
        if errors:
            print(f"Found {len(errors)} error(s):\n")
            for file, line, col, cat, msg in sorted(errors):
                print(f"  {file}:{line}:{col}: [{cat}] {msg}")
        if warnings:
            print(f"\nFound {len(warnings)} warning(s):\n")
            for file, line, col, cat, msg in sorted(warnings):
                print(f"  {file}:{line}:{col}: [{cat}] warning: {msg}")

    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tongues",
        description="Write algorithms once in Python. Get native code in every language.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify Python files against the Tongues subset",
    )
    verify_parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to verify",
    )
    verify_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    verify_parser.add_argument(
        "-c", "--by-category",
        action="store_true",
        help="Group violations by category",
    )
    verify_parser.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
