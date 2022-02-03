import os

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from dvclive.data.scalar import Scalar
from dvclive.lightning import DvcLiveLogger
from tests.test_main import read_logs

# pylint: disable=redefined-outer-name, unused-argument


class LitMNIST(LightningModule):
    def __init__(self):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, *args, **kwargs):
        x = args[0]
        batch_size, _, _, _ = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def train_dataloader(self):
        # transforms
        # prepare transforms standard to MNIST
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        # data
        mnist_train = MNIST(
            os.getcwd(), train=True, download=True, transform=transform
        )
        return DataLoader(mnist_train, batch_size=64)

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
    model = LitMNIST()
    # init logger
    dvclive_logger = DvcLiveLogger("test_run", path="logs")
    trainer = Trainer(
        logger=dvclive_logger, max_epochs=1, checkpoint_callback=False
    )
    trainer.fit(model)

    assert os.path.exists("logs")
    assert not os.path.exists("DvcLiveLogger")

    logs, _ = read_logs(tmp_dir / "logs" / Scalar.subfolder)

    assert len(logs) == 3
    assert "train_loss_step" in logs
    assert "train_loss_epoch" in logs
    assert "epoch" in logs
