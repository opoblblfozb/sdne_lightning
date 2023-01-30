from typing import Iterable, Sequence, Set

from torch.utils.data import DataLoader, Dataset
import torch


class Preprocessor:
    def __init__(self, edge: Iterable[Sequence]):
        self.edge = edge

    def get_node_size(self):
        set_data = self.get_node_set()
        return len(set_data)

    def get_node_set(self):
        node_set = set()
        for d in self.edge:
            for n in d:
                node_set.add(n)
        return node_set

    def create_dataset(self):
        return NetworkDataTorchDataset(self.edge, self.get_node_set())

    def create_data_loader(self, batch_size, shuffle):
        dataset = self.create_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class NetworkDataTorchDataset(Dataset):
    def __init__(self, edges: Iterable[Sequence[int]], nodes: Set):
        self.edges = edges
        self.nodes = nodes
        self.check_data()
        self.adj_matrix = self.create_adj_matrix()

    def check_data(self):
        sorted_nodes = sorted(self.nodes)
        for i, node in enumerate(sorted_nodes, start=1):
            if node != i:
                raise ValueError("nodesは、1からN(ノード数)までの連番で入力してください。")

    def create_adj_matrix(self) -> torch.Tensor:
        adj_matrix = torch.zeros(
            len(self.nodes), len(self.nodes), dtype=torch.float32
        )
        for pair in self.edges:
            adj_matrix[pair[0] - 1, pair[1] - 1] = 1
            adj_matrix[pair[1] - 1, pair[0] - 1] = 1
        return adj_matrix

    def __getitem__(self, index: int):
        return self.adj_matrix[index], index

    def __len__(self) -> int:
        return self.adj_matrix.shape[0]
