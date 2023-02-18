import numpy as np
import umap


class DimensionalityReduction:
    def __init__(self, embeddings: list[np.array]):
        self.embeddings = embeddings

    def features_UMAP(self, n_neighbors: int = 15, n_components: int = 5, metric: str = 'cosine') -> list[np.array]:
        return umap.UMAP(n_neighbors=n_neighbors,
                         n_components=n_components,
                         metric=metric).fit_transform(self.embeddings)

    # TODO PCA
    # TODO SVD
