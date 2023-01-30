import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
from torch.optim import Adam

import torch


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = SDNE(**ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["state_dict"])
    return model


class SDNE(pl.LightningModule):
    def __init__(self, node_size, beta, alpha, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.node_size = node_size
        self.beta = beta
        self.alpha = alpha
        self.encode0 = nn.Linear(node_size, 128)
        self.encode1 = nn.Linear(128, 64)
        self.decode0 = nn.Linear(64, 128)
        self.decode1 = nn.Linear(128, node_size)

    def forward(self, adj_batch):
        out = F.leaky_relu(self.encode0(adj_batch))
        embedding = F.leaky_relu(self.encode1(out))
        out = F.leaky_relu(self.decode0(embedding))
        out = F.leaky_relu(self.decode1(out))
        return out, embedding

    def training_step(self, batch, _):
        adj_batch, index = batch
        output, embedding = self(adj_batch)
        b_mat = torch.ones_like(adj_batch)
        b_mat[adj_batch != 0] = self.beta
        adj_mat = adj_batch[:, index]
        L_1st = self.calulate_loss_1(embedding, adj_mat)
        L_2st = self.caluculte_loss_2(output, adj_batch, b_mat)
        return {
            "loss": self.alpha * L_2st + L_1st,
            "loss_first": L_1st,
            "loss_second": L_2st,
        }

    def calulate_loss_1(self, embedding, adj_mat):
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)
        L_1st = torch.sum(
            adj_mat
            * (
                embedding_norm
                - 2
                * torch.mm(
                    embedding, torch.transpose(embedding, dim0=0, dim1=1)
                )
                + torch.transpose(embedding_norm, dim0=0, dim1=1)
            )
        )
        return L_1st

    def caluculte_loss_2(self, output, adj_batch, b_mat):
        return torch.sum(
            ((output - adj_batch) * b_mat) * ((output - adj_batch) * b_mat)
        )

    def training_epoch_end(self, train_step_outputs):
        all_loss = self.summary_loss("loss", train_step_outputs)
        loss_first = self.summary_loss("loss_first", train_step_outputs)
        loss_second = self.summary_loss("loss_second", train_step_outputs)
        self.log_dict(
            {
                "all_loss": all_loss,
                "loss_first": loss_first,
                "loss_second": loss_second,
            },
            prog_bar=True,
            logger=True,
        )
        print(
            "-------- Current Epoch {} --------".format(self.current_epoch + 1)
        )

    def summary_loss(self, loss_name, train_step_outputs):
        return sum([val[loss_name] for val in train_step_outputs]) / len(
            train_step_outputs
        )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-5)

        return optimizer
