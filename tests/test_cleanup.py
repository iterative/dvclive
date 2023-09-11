import pytest

from dvclive import Live
from dvclive.plots import Metric


@pytest.mark.parametrize(
    "html",
    [True, False],
)
@pytest.mark.parametrize(
    "dvcyaml",
    ["dvc.yaml", "logs/dvc.yaml"],
)
def test_cleanup(tmp_dir, html, dvcyaml):
    dvclive = Live("logs", report="html" if html else None, dvcyaml=dvcyaml)
    dvclive.log_metric("m1", 1)
    dvclive.next_step()

    html_path = tmp_dir / dvclive.dir / "report.html"
    if html:
        html_path.touch()

    (tmp_dir / "logs" / "some_user_file.txt").touch()
    (tmp_dir / "dvc.yaml").touch()

    assert (tmp_dir / dvclive.plots_dir / Metric.subfolder / "m1.tsv").is_file()
    assert (tmp_dir / dvclive.metrics_file).is_file()
    assert (tmp_dir / dvclive.dvc_file).is_file()
    assert html_path.is_file() == html

    dvclive = Live("logs")

    assert (tmp_dir / "logs" / "some_user_file.txt").is_file()
    assert not (tmp_dir / dvclive.plots_dir / Metric.subfolder).exists()
    assert not (tmp_dir / dvclive.metrics_file).is_file()
    if dvcyaml == "dvc.yaml":
        assert (tmp_dir / dvcyaml).is_file()
    if dvcyaml == "logs/dvc.yaml":
        assert not (tmp_dir / dvcyaml).is_file()
    assert not (html_path).is_file()
