import pytest


@pytest.fixture
def tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path
