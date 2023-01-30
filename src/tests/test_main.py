import sdne_lightning

from unittest import TestCase
from unittest.mock import patch


class MainTestCase(TestCase):
    @patch("sdne_lightning.ArgumentParser")
    @patch("sdne_lightning.train_sdne")
    def test_train(self, train_sdne, parser):
        options = parser.return_value.parse_args.return_value
        options.subcommand = "train"
        options.input_graph = "input.tsv"
        options.train_config = "config.json"
        options.model_path = "model/dir/"

        sdne_lightning.main()

        train_sdne.assert_called_once_with(
            input_graph="input.tsv",
            train_config="config.json",
            model_path="model/dir/",
        )
