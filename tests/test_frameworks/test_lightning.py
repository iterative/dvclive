import os

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from dvclive.lightning import DVCLiveLogger
from dvclive.plots.metric import Metric
from dvclive.serialize import load_yaml
from dvclive.utils import parse_metrics


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
    def __init__(self, latent_dims=4):
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
        x = F.log_softmax(x, dim=1)
        return x

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
        return Adam(self.parameters(), lr=1e-3)

    def predict_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass


def test_lightning_integration(tmp_dir, mocker):
    # init model
    model = LitXOR()
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
    assert load_yaml(params_file) == {"latent_dims": 4}


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
    dvclive_logger = DVCLiveLogger(dir="dir", report="md", dvcyaml=False)
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
    dvclive_logger = DVCLiveLogger(save_dvc_exp=True)
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
