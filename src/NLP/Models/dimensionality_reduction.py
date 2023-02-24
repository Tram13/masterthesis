import numpy as np
import pandas as pd
import umap


class DimensionalityReduction:
    def __init__(self, embeddings: pd.DataFrame) -> None:
        self.embeddings = embeddings

    def get_UMAP_model(self, n_neighbors: int = 30, n_components: int = 15, metric: str = 'cosine'):
        return umap.UMAP(n_neighbors=n_neighbors,
                         n_components=n_components,
                         metric=metric)

    # TODO PCA
    # TODO TruncatedSVD

    def features_UMAP(self, n_neighbors: int = 30, n_components: int = 15, metric: str = 'cosine') -> pd.DataFrame:
        return pd.DataFrame(self.get_UMAP_model(n_neighbors=n_neighbors,
                                                n_components=n_components,
                                                metric=metric).fit_transform(self.embeddings))


