from sdne_lightning.model import SDNE
from sdne_lightning.data import NetworkData
from sdne_lightning.util import read_tsv_as_int_list

import pytorch_lightning as pl


def prepare_trainer():
    return pl.Trainer(max_epochs=1000)


def train_sdne(input_graph, model_path):
    data = NetworkData(read_tsv_as_int_list(input_graph))
    model = SDNE(data.get_node_size())
    trainer = prepare_trainer()
    trainer.fit(
        model=model,
        train_dataloaders=data.create_data_loader(batch_size=8, shuffle=True),
    )
