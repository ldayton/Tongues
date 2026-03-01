"""Compatibility shim — re-exports tests from per-group files.

The justfile targets reference test_runner.py with -k filters.
This file re-exports all test functions and pytest_generate_tests
so those targets keep working until the justfile is updated (step 4).
"""

from tests.test_frontend import *  # noqa: F401,F403
from tests.test_middleend import *  # noqa: F401,F403
from tests.test_codegen import *  # noqa: F401,F403
from tests.test_target import *  # noqa: F401,F403
from tests.test_ty_app import *  # noqa: F401,F403

from tests.harness import EMITTERS, TESTS_DIR

from tests.test_frontend import TESTS as _FRONTEND_TESTS
from tests.test_middleend import TESTS as _MIDDLEEND_TESTS
from tests.test_codegen import TESTS as _CODEGEN_TESTS
from tests.test_target import TESTS as _TARGET_TESTS
from tests.test_ty_app import TESTS as _TY_APP_TESTS

# Merged TESTS dict for pytest_generate_tests
TESTS = {
    "cli": {"cli": _FRONTEND_TESTS["cli"]},
    "frontend": {k: v for k, v in _FRONTEND_TESTS.items() if k != "cli"},
    "middleend": dict(_MIDDLEEND_TESTS),
    "backend": dict(_CODEGEN_TESTS) | dict(_TARGET_TESTS),
    "taytsh": dict(_TY_APP_TESTS),
}


def pytest_generate_tests(metafunc):
    """Dispatch to per-group parametrize logic."""
    from tests.test_frontend import pytest_generate_tests as _fe
    from tests.test_middleend import pytest_generate_tests as _me
    from tests.test_codegen import pytest_generate_tests as _cg
    from tests.test_target import pytest_generate_tests as _tg
    from tests.test_ty_app import pytest_generate_tests as _ta

    _fe(metafunc)
    _me(metafunc)
    _cg(metafunc)
    _tg(metafunc)
    _ta(metafunc)
