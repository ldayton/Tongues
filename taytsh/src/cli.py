"""Taytsh CLI — parse and run .ty files."""

from __future__ import annotations

import os
import sys

from . import parse
from .runtime import TaytshError, TaytshRuntimeFault, TaytshTypeError, run


USAGE: str = """\
taytsh [OPTIONS] FILE

Run a Taytsh (.ty) program.

Options:
  --strict           Enable --strict-math and --strict-tostring
  --strict-math      Enable strict math mode
  --strict-tostring  Enable strict tostring mode
  --help             Show this help message
"""


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    filepath: str = ""
    strict = False
    strict_math = False
    strict_tostring = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--help" or arg == "-h":
            print(USAGE, end="")
            return 0
        elif arg == "--strict":
            strict = True
            i += 1
        elif arg == "--strict-math":
            strict_math = True
            i += 1
        elif arg == "--strict-tostring":
            strict_tostring = True
            i += 1
        elif arg.startswith("-"):
            print("taytsh: unknown flag '" + arg + "'", file=sys.stderr)
            return 2
        elif filepath == "":
            filepath = arg
            i += 1
        else:
            print("taytsh: unexpected argument '" + arg + "'", file=sys.stderr)
            return 2
    if filepath == "":
        print("taytsh: missing file argument", file=sys.stderr)
        return 2

    try:
        with open(filepath, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        print("taytsh: " + filepath + ": No such file or directory", file=sys.stderr)
        return 1
    except OSError as e:
        print("taytsh: " + filepath + ": " + str(e), file=sys.stderr)
        return 1
    try:
        source = raw.decode("utf-8")
    except ValueError:
        print("taytsh: " + filepath + ": invalid utf-8", file=sys.stderr)
        return 1

    try:
        module = parse(source)
    except Exception as e:
        print("taytsh: parse error: " + str(e), file=sys.stderr)
        return 1

    if strict or strict_math:
        module.strict_math = True
    if strict or strict_tostring:
        module.strict_tostring = True

    try:
        result = run(
            module,
            stdin=sys.stdin.buffer.read() if not sys.stdin.isatty() else b"",
            args=sys.argv[1:],
            env=dict(os.environ),
        )
    except TaytshTypeError as e:
        print("taytsh: type error: " + str(e), file=sys.stderr)
        return 1
    except TaytshRuntimeFault as e:
        print("taytsh: runtime error: " + str(e), file=sys.stderr)
        return 1
    except TaytshError as e:
        print("taytsh: error: " + str(e), file=sys.stderr)
        return 1

    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
