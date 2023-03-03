from sklearn.cluster import KMeans
import pandas as pd


class ClusteringAlgorithms:
    def __init__(self, features: pd.DataFrame) -> None:
        self.features = features

    def get_KMEANS_Model(self, init: str = "random", n_clusters: int = 10, n_init: int = 10, max_iter: int = 300):
        return KMeans(
            init=init,              # initialization of cluster centers
            n_clusters=n_clusters,  # amount of clusters
            n_init=n_init,          # number of times the algorithm is run with cluster centers
            max_iter=max_iter,      # maximum amount of iterations
            random_state=42         # random state for reproduction
        )

    def labels_KMEANS(self, init: str = "random", n_clusters: int = 10, n_init: int = 10,
                      max_iter: int = 300) -> tuple[pd.Series, int]:
        kmeans = self.get_KMEANS_Model(
            init=init,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
        )
        clustered = kmeans.fit(self.features)
        return pd.Series(clustered.labels_, name='cluster_labels'), n_clusters

