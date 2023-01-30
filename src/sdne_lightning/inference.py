class Inferencer:
    def __init__(self, model):
        self.model = model

    def embed(self, adj_matrix):
        reconstruction, embedding = self.model(adj_matrix)
        return embedding.tolist()
