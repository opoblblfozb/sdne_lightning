import numpy as np
import json


def read_tsv_as_int_list(file_path: str) -> list[list]:
    return np.loadtxt(file_path, delimiter="\t", dtype=np.int64).tolist()


def read_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        res = json.load(f)
    return res
