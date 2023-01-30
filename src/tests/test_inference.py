from unittest import TestCase
from unittest.mock import MagicMock

from sdne_lightning.inference import Inferencer
from sdne_lightning.model import SDNE

import torch
import numpy as np


class InferencerTestCase(TestCase):
    def test__init__(self):
        model = MagicMock(spec=SDNE)

        sut = Inferencer(model)

        self.assertEqual(sut.model, model)

    def test_inference(self):
        model = MagicMock(spec=SDNE)
        adj_matrix = [[0, 0, 1], [1, 0, 0], [1, 0, 0]]
        reconstruction = torch.tensor(
            [[0, 0, 1], [1, 0, 0], [1, 0, 0]], dtype=torch.float32
        )
        embedding = torch.tensor(
            [[0.5, 0.4], [0.4, 0.4], [0.1, 0.2]], dtype=torch.float32
        )
        model.return_value = reconstruction, embedding

        sut = Inferencer(model)
        actual = sut.embed(adj_matrix)

        self.assertAlmostEqual(actual, [[0.5, 0.4], [0.4, 0.4], [0.1, 0.2]])

    def assertAlmostEqual(self, first, second):
        first_arr = np.array(first)
        second_arr = np.array(second)
        np.testing.assert_allclose(first_arr, second_arr)
