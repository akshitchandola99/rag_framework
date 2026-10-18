"""Phase 8: settings shared by every test folder."""

from pathlib import Path

import pytest

TESTS_FOLDER = Path(__file__).resolve().parent
SLOW_TEST_FOLDERS = [TESTS_FOLDER / "integration", TESTS_FOLDER / "api"]
FAST_RUN_MARKER_EXPRESSION = "not integration"


# True when a path is inside tests/integration or tests/api.
def is_in_slow_folder(path: Path) -> bool:
    for slow_folder in SLOW_TEST_FOLDERS:
        if path.is_relative_to(slow_folder):
            return True
    return False


# Mark every test in tests/integration and tests/api as integration, even if its file forgot the marker.
def pytest_collection_modifyitems(config, items):
    for item in items:
        if is_in_slow_folder(item.path):
            item.add_marker(pytest.mark.integration)


# On a -m "not integration" run, don't even import the slow folders: every test there would be deselected anyway.
def pytest_ignore_collect(collection_path, config):
    if config.getoption("markexpr") != FAST_RUN_MARKER_EXPRESSION:
        return None
    if is_in_slow_folder(collection_path):
        return True
    return None
