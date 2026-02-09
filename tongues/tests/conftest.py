"""Pytest configuration for Tongues test suite."""

import sys
from pathlib import Path

import pytest

# Add tongues directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

APP_DIR = Path(__file__).parent / "15_app"


def pytest_addoption(parser):
    """Add --target, --ignore-version, --ignore-skips, and --summary options."""
    parser.addoption(
        "--target",
        action="append",
        default=[],
        help="Run only specified targets (can be used multiple times)",
    )
    parser.addoption(
        "--ignore-version",
        action="store_true",
        default=False,
        help="Skip version checks and run tests with whatever runtime is available",
    )
    parser.addoption(
        "--ignore-skips",
        action="store_true",
        default=False,
        help="Run tests even if they are in the known-failure skip list",
    )
    parser.addoption(
        "--summary",
        action="store_true",
        default=False,
        help="Print a summary table of apptest results",
    )


# --- Summary table support ---

_apptest_results: dict[
    str, dict[str, tuple[int, int]]
] = {}  # {lang: {test: (passed, total)}}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture apptest results for summary table."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or "apptest" not in item.fixturenames:
        return
    if "[" not in item.nodeid:
        return
    test_id = item.nodeid.split("[")[1].rstrip("]")
    if "/" not in test_id:
        return
    lang, test_name = test_id.split("/", 1)
    if lang not in _apptest_results:
        _apptest_results[lang] = {}
    output = ""
    if report.failed and report.longrepr:
        output = str(report.longrepr)
    passed = output.count("PASS test_")
    failed = output.count("FAIL test_")
    total = passed + failed
    if total == 0 and report.passed:
        apptest_file = APP_DIR / f"{test_name}.py"
        if apptest_file.exists():
            total = apptest_file.read_text().count("\ndef test_")
            passed = total
    _apptest_results[lang][test_name] = (passed, total)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print summary table if --summary flag is set."""
    if not config.getoption("summary") or not _apptest_results:
        return
    terminalreporter.write_sep("=", "Apptest Summary")
    all_tests = sorted(
        {t for lang_results in _apptest_results.values() for t in lang_results}
    )
    langs = sorted(_apptest_results.keys())
    header = f"{'Test':<20}" + "".join(f"{lang:<15}" for lang in langs)
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * len(header))
    for test in all_tests:
        row = f"{test:<20}"
        for lang in langs:
            if test in _apptest_results.get(lang, {}):
                passed, total = _apptest_results[lang][test]
                if passed == total and total > 0:
                    cell = f"✅ {passed}/{total}"
                else:
                    cell = f"❌ {passed}/{total}"
            else:
                cell = "-"
            row += f"{cell:<15}"
        terminalreporter.write_line(row)
