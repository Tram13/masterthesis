import numpy as np
import pandas as pd
import umap


class DimensionalityReduction:
    def __init__(self, embeddings: pd.DataFrame) -> None:
        self.embeddings = embeddings

    def features_UMAP(self, n_neighbors: int = 30, n_components: int = 15, metric: str = 'cosine') -> pd.DataFrame:
        return pd.DataFrame(umap.UMAP(n_neighbors=n_neighbors,
                                      n_components=n_components,
                                      metric=metric).fit_transform(self.embeddings))

    # TODO PCA
    # TODO SVD
