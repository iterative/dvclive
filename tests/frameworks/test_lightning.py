import os

import pytest

from dvclive.plots.metric import Metric
from dvclive.serialize import load_yaml
from dvclive.utils import parse_metrics

try:
    import torch
    from lightning import LightningModule
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch import nn
    from torch.nn import functional as F  # noqa: N812
    from torch.optim import SGD, Adam
    from torch.utils.data import DataLoader, Dataset

    from dvclive import Live
    from dvclive.lightning import DVCLiveLogger
except ImportError:
    pytest.skip("skipping lightning tests", allow_module_level=True)


class XORDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.ins = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.outs = [1, 0, 0, 1]

    def __getitem__(self, index):
        return torch.Tensor(self.ins[index]), torch.tensor(
            self.outs[index], dtype=torch.long
        )

    def __len__(self):
        return len(self.ins)


class LitXOR(LightningModule):
    def __init__(
        self,
        latent_dims=4,
        optim=SGD,
        optim_params={"lr": 0.01},  # noqa: B006
        input_size=[256, 256, 256],  # noqa: B006
    ):
        super().__init__()

        self.save_hyperparameters()

        self.layer_1 = nn.Linear(2, latent_dims)
        self.layer_2 = nn.Linear(latent_dims, 2)

    def forward(self, *args, **kwargs):
        x = args[0]
        batch_size, _ = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return F.log_softmax(x, dim=1)

    def train_loader(self):
        dataset = XORDataset()
        return DataLoader(dataset, batch_size=1)

    def train_dataloader(self):
        return self.train_loader()

    def training_step(self, *args, **kwargs):
        batch = args[0]
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        return self.hparams.optim(self.parameters(), **self.hparams.optim_params)

    def predict_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass


def test_lightning_integration(tmp_dir, mocker):
    # init model
    model = LitXOR(
        latent_dims=8, optim=Adam, optim_params={"lr": 0.02}, input_size=[128, 128, 128]
    )
    # init logger
    dvclive_logger = DVCLiveLogger("test_run", dir="logs")
    live = dvclive_logger.experiment
    spy = mocker.spy(live, "end")
    trainer = Trainer(
        logger=dvclive_logger,
        max_epochs=2,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    trainer.fit(model)
    spy.assert_called_once()

    assert os.path.exists("logs")
    assert not os.path.exists("DvcLiveLogger")

    scalars = os.path.join(dvclive_logger.experiment.plots_dir, Metric.subfolder)
    logs, _ = parse_metrics(dvclive_logger.experiment)

    assert len(logs) == 3
    assert os.path.join(scalars, "train", "epoch", "loss.tsv") in logs
    assert os.path.join(scalars, "train", "step", "loss.tsv") in logs
    assert os.path.join(scalars, "epoch.tsv") in logs

    params_file = dvclive_logger.experiment.params_file
    assert os.path.exists(params_file)
    assert load_yaml(params_file) == {
        "latent_dims": 8,
        "optim": "Adam",
        "optim_params": {"lr": 0.02},
        "input_size": [128, 128, 128],
    }


def test_lightning_default_dir(tmp_dir):
    model = LitXOR()
    # If `dir` is not provided handle it properly, use default value
    dvclive_logger = DVCLiveLogger("test_run")
    trainer = Trainer(
        logger=dvclive_logger,
        max_epochs=2,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    trainer.fit(model)

    assert os.path.exists("dvclive")


def test_lightning_kwargs(tmp_dir):
    model = LitXOR()
    # Handle kwargs passed to Live.
    dvclive_logger = DVCLiveLogger(
        dir="dir", report="md", dvcyaml=False, cache_images=True
    )
    trainer = Trainer(
        logger=dvclive_logger,
        max_epochs=2,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    trainer.fit(model)

    assert os.path.exists("dir")
    assert os.path.exists("dir/report.md")
    assert not os.path.exists("dir/dvc.yaml")
    assert dvclive_logger.experiment._cache_images is True


@pytest.mark.parametrize("log_model", [False, True, "all"])
@pytest.mark.parametrize("save_top_k", [1, -1])
def test_lightning_log_model(tmp_dir, mocker, log_model, save_top_k):
    model = LitXOR()
    dvclive_logger = DVCLiveLogger(dir="dir", log_model=log_model)
    checkpoint = ModelCheckpoint(dirpath="model", save_top_k=save_top_k)
    trainer = Trainer(
        logger=dvclive_logger,
        max_epochs=2,
        log_every_n_steps=1,
        callbacks=[checkpoint],
    )
    log_artifact = mocker.patch.object(dvclive_logger.experiment, "log_artifact")
    trainer.fit(model)

    # Check that log_artifact is called.
    if log_model is False:
        log_artifact.assert_not_called()
    elif (log_model is True) and (save_top_k != -1):
        # called once to cache, then again to log best artifact
        assert log_artifact.call_count == 2
    else:
        # once per epoch plus two calls at the end (see above)
        assert log_artifact.call_count == 4

    # Check that checkpoint files does not grow with each run.
    num_checkpoints = len(os.listdir(tmp_dir / "model"))
    if log_model in [True, "all"]:
        trainer.fit(model)
        assert len(os.listdir(tmp_dir / "model")) == num_checkpoints
        log_artifact.assert_any_call(
            checkpoint.best_model_path, name="best", type="model", copy=True
        )


def test_lightning_steps(tmp_dir, mocker):
    model = LitXOR()
    # Handle kwargs passed to Live.
    dvclive_logger = DVCLiveLogger(dir="logs")
    live = dvclive_logger.experiment
    spy = mocker.spy(live, "next_step")
    trainer = Trainer(
        logger=dvclive_logger,
        max_epochs=2,
        enable_checkpointing=False,
        # Log one time in the middle of the epoch
        log_every_n_steps=3,
    )
    trainer.fit(model)

    history, latest = parse_metrics(dvclive_logger.experiment)
    assert latest["step"] == 7
    assert latest["epoch"] == 1

    scalars = os.path.join(dvclive_logger.experiment.plots_dir, Metric.subfolder)
    epoch_loss = history[os.path.join(scalars, "train", "epoch", "loss.tsv")]
    step_loss = history[os.path.join(scalars, "train", "step", "loss.tsv")]
    assert len(epoch_loss) == 2
    assert len(step_loss) == 2

    # call next_step:
    # - 2x epoch end
    # - 2x log_every_n_steps
    assert spy.call_count == 4


class ValLitXOR(LitXOR):
    def val_loader(self):
        dataset = XORDataset()
        return DataLoader(dataset, batch_size=1)

    def val_dataloader(self):
        return self.val_loader()

    def training_step(self, *args, **kwargs):
        batch = args[0]
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, *args, **kwargs):
        batch = args[0]
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss


def test_lightning_val_udpates_to_studio(tmp_dir, mocked_dvc_repo, mocked_studio_post):
    """Test the `self.experiment._latest_studio_step -= 1` logic."""
    mocked_post, _ = mocked_studio_post

    model = ValLitXOR()
    dvclive_logger = DVCLiveLogger()
    trainer = Trainer(
        logger=dvclive_logger,
        max_steps=4,
        val_check_interval=2,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    trainer.fit(model)

    calls = mocked_post.call_args_list
    # 0: start
    # 1: update_train_step_metrics
    # 2: update_train_step_metrics
    # 3: log_eval_end_metrics
    plots = calls[3][1]["json"]["plots"]
    val_loss = plots["dvclive/dvc.yaml::dvclive/plots/metrics/val/loss.tsv"]
    # Without `self.experiment._latest_studio_step -= 1`
    # This would be empty
    assert len(val_loss["data"]) == 1


def test_lightning_force_init(tmp_dir, mocker):
    """Regression test for https://github.com/iterative/dvclive/issues/594
    Only call Live.__init__ when report is notebook.
    """
    init = mocker.spy(Live, "__init__")
    DVCLiveLogger()
    init.assert_not_called()
    DVCLiveLogger(report="notebook")
    init.assert_called_once()
