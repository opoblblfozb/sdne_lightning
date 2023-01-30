from sdne_lightning.model import SDNE
from sdne_lightning.preprocessor import Preprocessor
from sdne_lightning.util import read_tsv_as_int_list, read_json

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


def prepare_trainer():
    return pl.Trainer(max_epochs=500, logger=TensorBoardLogger("./logs"))


def train_sdne(input_graph, train_config, model_path):
    data = Preprocessor(read_tsv_as_int_list(input_graph))
    model = SDNE(node_size=data.get_node_size(), **read_json(train_config))
    trainer = prepare_trainer()
    trainer.fit(
        model=model,
        train_dataloaders=data.create_data_loader(batch_size=8, shuffle=True),
    )
