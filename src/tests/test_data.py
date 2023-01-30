from sdne_lightning.preprocessor import Preprocessor

from unittest import TestCase
from unittest.mock import MagicMock, patch


class NetworkDataTestCase(TestCase):
    MODULE_UNDER_TEST = "sdne_lightning.preprocessor"

    def test_init(self):
        data = [[1, 2], [3, 4]]

        sut = Preprocessor(data)

        self.assertEqual(sut.edge, data)

    def test_get_node_set(self):
        data = [[1, 2], [3, 4]]

        sut = Preprocessor(data)

        self.assertEqual(sut.get_node_set(), {1, 2, 3, 4})

    @patch(f"{MODULE_UNDER_TEST}.Preprocessor.get_node_set")
    def test_get_node_size(self, get_node_set):
        data = [[1, 2], [3, 4]]
        node_set = get_node_set.return_value

        sut = Preprocessor(data)
        actual = sut.get_node_size()

        get_node_set.assert_called_once_with()
        node_set.__len__.assert_called_once_with()
        self.assertEqual(actual, node_set.__len__.return_value)

    @patch(f"{MODULE_UNDER_TEST}.DataLoader")
    @patch(f"{MODULE_UNDER_TEST}.Preprocessor.create_dataset")
    def test_create_train_dataset(self, create_dataset, DataLoader):
        sut = Preprocessor(MagicMock())

        actual = sut.create_data_loader(batch_size=64, shuffle=True)

        create_dataset.assert_called_once_with()
        DataLoader.assert_called_once_with(
            create_dataset.return_value, batch_size=64, shuffle=True
        )

    @patch(f"{MODULE_UNDER_TEST}.NetworkDataTorchDataset")
    @patch(f"{MODULE_UNDER_TEST}.Preprocessor.get_node_set")
    def test_create_dataset(self, get_node_set, torch_dataset):
        sut = Preprocessor(MagicMock())

        dataset = sut.create_dataset()

        torch_dataset.assert_called_once_with(
            sut.edge, get_node_set.return_value
        )


from sdne_lightning.preprocessor import NetworkDataTorchDataset
from torch.testing import assert_allclose
import torch


class NetworkDataTorchDatasetTestCase(TestCase):
    def test_create_adj_matrix(self):
        edges = [[1, 2], [2, 3]]
        nodes = {1, 2, 3}

        sut = NetworkDataTorchDataset(edges, nodes)
        actual = sut.create_adj_matrix()

        expected = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        assert_allclose(actual, expected)

    def test_check_data_when_invalid(self):
        edges = [[1, 2], [2, 3]]
        nodes = {3, 5, 6}

        with self.assertRaises(ValueError) as cm:
            sut = NetworkDataTorchDataset(edges, nodes)
        self.assertEqual(
            cm.exception.args[0], "nodesは、1からN(ノード数)までの連番で入力してください。"
        )
