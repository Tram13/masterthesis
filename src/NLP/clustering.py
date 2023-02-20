from sklearn.cluster import KMeans
import pandas as pd


class ClusteringAlgorithms:
    def __init__(self, features: pd.DataFrame) -> None:
        self.features = features

    def labels_KMEANS(self, init: str = "random", n_clusters: int = 10, n_init: int = 10, max_iter: int = 300) -> pd.Series:
        kmeans = KMeans(
                        init=init,                  # initialization of cluster centers
                        n_clusters=n_clusters,      # amount of clusters
                        n_init=n_init,              # number of times the algorithm is run with cluster centers
                        max_iter=max_iter,          # maximum amount of iterations
                        random_state=42             # random state for reproduction
        )
        clustered = kmeans.fit(self.features)
        return pd.Series(clustered.labels_, name='cluster_labels')


    # TODO PCA
    # TODO SVD
