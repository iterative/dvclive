import os

import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from dvclive.lightning import DVCLiveLogger
from dvclive.plots.metric import Metric
from dvclive.utils import parse_metrics

# pylint: disable=redefined-outer-name, unused-argument


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
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(2, 4)
        self.layer_2 = nn.Linear(4, 2)

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
        loader = DataLoader(dataset, batch_size=1)
        return loader

    def train_dataloader(self):
        loader = self.train_loader()
        return loader

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


def test_lightning_integration(tmp_dir):
    # init model
    model = LitXOR()
    # init logger
    dvclive_logger = DVCLiveLogger("test_run", dir="logs")
    trainer = Trainer(
        logger=dvclive_logger,
        max_epochs=2,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    trainer.fit(model)

    assert os.path.exists("logs")
    assert not os.path.exists("DvcLiveLogger")

    scalars = os.path.join(
        dvclive_logger.experiment.plots_dir, Metric.subfolder
    )
    logs, _ = parse_metrics(dvclive_logger.experiment)

    assert len(logs) == 3
    assert os.path.join(scalars, "train", "epoch", "loss.tsv") in logs
    assert os.path.join(scalars, "train", "step", "loss.tsv") in logs
    assert os.path.join(scalars, "epoch.tsv") in logs


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
