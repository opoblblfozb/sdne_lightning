# %%
from sdne_lightning.model import SDNE
from pathlib import Path

current = Path(__file__).parent.resolve()
file = (
    current.parent
    / "logs"
    / "lightning_logs"
    / "version_2"
    / "checkpoints"
    / "epoch=499-step=500.ckpt"
)
params = (
    current.parent
    / "logs"
    / "lightning_logs"
    / "version_2"
    / "checkpoints"
    / "hparams.yaml"
)
import torch

ckpt = torch.load(file)
model = SDNE(**ckpt["hyper_parameters"])
model.load_state_dict(ckpt["state_dict"])
model.eval()


# %%
import torch

k = torch.zeros(8, 8)
k[0, 0] = 1

# %%

model(k)

# %%
