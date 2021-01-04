import pytest


@pytest.fixture
def tmp_dir(tmp_path, monkeypatch):
    import dvclive

    dvclive._metric_logger = None  # pylint: disable=protected-access
    monkeypatch.chdir(tmp_path)
    yield tmp_path
