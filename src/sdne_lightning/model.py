import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam

import torch


class SDNE(pl.LightningModule):
    def __init__(self, node_size):
        super().__init__()
        self.node_size = node_size
        self.encode0 = nn.Linear(node_size, 128)
        self.encode1 = nn.Linear(128, 64)
        self.decode0 = nn.Linear(64, 128)
        self.decode1 = nn.Linear(128, node_size)

    def forward(self, adj_batch):
        out = self.encode0(adj_batch)
        out = self.encode1(out)
        out = self.decode0(out)
        out = self.decode1(out)
        return out

    def training_step(self, adj_batch, _):
        output = self(adj_batch)
        loss = torch.sum((output - adj_batch) * (output - adj_batch))
        return {
            "loss": loss,
        }

    def training_epoch_end(self, train_step_outputs):
        step_num = len(train_step_outputs)
        epoch_loss = (
            sum([val["loss"] for val in train_step_outputs]) / step_num
        )

        self.log(
            "train_loss", epoch_loss, prog_bar=True, on_epoch=True, logger=True
        )

        print(
            "-------- Current Epoch {} --------".format(self.current_epoch + 1)
        )
        print("train Loss: {:.4f}".format(epoch_loss))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-5)

        return optimizer
