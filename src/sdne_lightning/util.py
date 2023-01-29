import numpy as np


def read_tsv_as_int_list(file_path: str) -> list[list]:
    return np.loadtxt(file_path, delimiter="\t", dtype=np.int64).tolist()
