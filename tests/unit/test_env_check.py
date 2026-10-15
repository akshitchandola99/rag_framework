from ragqa import env_check


# Python must be 3.12 or newer.
def test_python_version_meets_minimum():
    assert env_check.check_python_version() is True


# Python must run as arm64, not under Rosetta.
def test_running_on_apple_silicon():
    assert env_check.check_architecture() is True