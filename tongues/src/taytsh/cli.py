"""Taytsh CLI — parse and run .ty files."""

from __future__ import annotations

import sys

from . import parse
from .ast import TModule
from .compiler import CompileError
from .treewalker import RunResult, TaytshRuntimeFault, TaytshTypeError, run
from .vm import VMResult, vm_run


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


def cli_main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    filepath: str = ""
    strict = False
    strict_math = False
    strict_tostring = False
    use_vm = False
    for arg in args:
        if arg == "--help" or arg == "-h":
            print(USAGE, end="")
            return 0
        elif arg == "--vm":
            use_vm = True
        elif arg == "--strict":
            strict = True
        elif arg == "--strict-math":
            strict_math = True
        elif arg == "--strict-tostring":
            strict_tostring = True
        elif arg.startswith("-"):
            print("taytsh: unknown flag '" + arg + "'", file=sys.stderr)
            return 2
        elif not filepath:
            filepath = arg
        else:
            print("taytsh: unexpected argument '" + arg + "'", file=sys.stderr)
            return 2
    if not filepath:
        print("taytsh: missing file argument", file=sys.stderr)
        return 2

    raw: bytes = b""
    source: str = ""
    module: TModule = TModule([], False, False)
    err_code = 0
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        source = raw.decode("utf-8")
        module = parse(source)
    except OSError as e:
        print("taytsh: " + filepath + ": " + str(e), file=sys.stderr)
        err_code = 1
    except ValueError:
        print("taytsh: " + filepath + ": invalid utf-8", file=sys.stderr)
        err_code = 1
    except Exception as e:
        print(str(e), file=sys.stderr)
        err_code = 1
    if err_code != 0:
        return err_code

    if strict or strict_math:
        module.strict_math = True
    if strict or strict_tostring:
        module.strict_tostring = True

    if use_vm:
        return _run_vm(module)
    return _run_interp(module)


def _run_vm(module: TModule) -> int:
    result: VMResult = VMResult(0, "", "")
    try:
        result = vm_run(
            module,
            stdin=sys.stdin.buffer.read(),
            args=sys.argv[1:],
        )
    except CompileError as e:
        print("taytsh: compile error: " + str(e), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(result.stdout.encode("utf-8"))
    sys.stderr.buffer.write(result.stderr.encode("utf-8"))
    return result.exit_code


def _run_interp(module: TModule) -> int:
    result: RunResult = RunResult(0, b"", b"")
    try:
        result = run(module, sys.stdin.buffer.read(), sys.argv[1:])
    except TaytshTypeError as e:
        print("taytsh: type error: " + str(e), file=sys.stderr)
        return 1
    except TaytshRuntimeFault as e:
        print("taytsh: runtime error: " + str(e), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(cli_main())
