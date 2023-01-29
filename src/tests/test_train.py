from sdne_lightning import train

from unittest import TestCase
from unittest.mock import patch


class TrainSDNETestCase(TestCase):
    MODLE_UNDER_TEST_PATH = "sdne_lightning.train"

    @patch(f"{MODLE_UNDER_TEST_PATH}.read_tsv_as_int_list")
    @patch(f"{MODLE_UNDER_TEST_PATH}.SDNE")
    @patch(f"{MODLE_UNDER_TEST_PATH}.prepare_trainer")
    @patch(f"{MODLE_UNDER_TEST_PATH}.NetworkData")
    def test_train_sdne(self, NetworkData, prepare_trainer, SDNE, read_tsv):
        data = NetworkData.return_value
        trainer = prepare_trainer.return_value

        train.train_sdne("input.tsv", "model/dir/")

        read_tsv.assert_called_once_with("input.tsv")
        NetworkData.assert_called_once_with(read_tsv.return_value)
        SDNE.assert_called_once_with(data.get_node_size.return_value)
        prepare_trainer.assert_called_once_with()
        trainer.fit.assert_called_once_with(
            model=SDNE.return_value,
            train_dataloaders=data.create_data_loader.return_value,
        )
