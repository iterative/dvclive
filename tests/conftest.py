import sys

import pytest


@pytest.fixture
def tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture(autouse=True)
def capture_wrap():
    # https://github.com/pytest-dev/pytest/issues/5502#issuecomment-678368525
    sys.stderr.close = lambda *args: None
    sys.stdout.close = lambda *args: None
    yield
