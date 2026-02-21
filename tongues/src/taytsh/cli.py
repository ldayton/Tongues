"""Taytsh CLI — parse and run .ty files."""  # tongues: skip

from __future__ import annotations

import os
import sys

from . import parse
from .compiler import CompileError
from .runtime import TaytshError, TaytshRuntimeFault, TaytshTypeError, run
from .vm import vm_run


USAGE: str = """\
taytsh [OPTIONS] FILE

Run a Taytsh (.ty) program.

Options:
  --vm               Run through the bytecode VM
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
    use_vm = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--help" or arg == "-h":
            print(USAGE, end="")
            return 0
        elif arg == "--vm":
            use_vm = True
            i += 1
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
        print(str(e), file=sys.stderr)
        return 1

    if strict or strict_math:
        module.strict_math = True
    if strict or strict_tostring:
        module.strict_tostring = True

    if use_vm:
        return _run_vm(module)
    return _run_interp(module)


def _run_vm(module) -> int:  # type: ignore[no-untyped-def]
    try:
        result = vm_run(
            module,
            stdin=sys.stdin.buffer.read() if not sys.stdin.isatty() else b"",
            args=sys.argv[1:],
            env=dict(os.environ),
        )
    except CompileError as e:
        print("taytsh: compile error: " + str(e), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(result.stdout.encode("utf-8"))
    sys.stderr.buffer.write(result.stderr.encode("utf-8"))
    return result.exit_code


def _run_interp(module) -> int:  # type: ignore[no-untyped-def]
    try:
        result = run(
            module,
            sys.stdin.buffer.read() if not sys.stdin.isatty() else b"",
            sys.argv[1:],
            dict(os.environ),
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
