from dvclive import dvclive, init


def test_create_logs_dir(tmp_dir):
    init("test_dir")

    assert (tmp_dir / "test_dir").is_dir()


def test_logging(tmp_dir):
    init("test_dir")

    dvclive.log("m1", 1)

    dvclive.next_step()

    assert (tmp_dir / "test_dir" / "history").is_dir()
    assert (tmp_dir / "test_dir" / "history" / "m1.tsv").is_file()
    assert (tmp_dir / "test_dir" / "latest.json").is_file()
