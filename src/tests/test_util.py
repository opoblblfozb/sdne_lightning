from unittest import TestCase

from sdne_lightning.util import read_tsv_as_int_list
from pathlib import Path

import numpy as np


def test_read_tsv_as_int_list():
    current = Path(__file__).parent.resolve()
    fixture = current / "resource" / "dummy.tsv"
    actual = read_tsv_as_int_list(fixture)
    assert actual == [[1, 2], [3, 4]]
