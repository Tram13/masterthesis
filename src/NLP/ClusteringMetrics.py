import numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.libs.jqmcvi_base import dunn_fast


class ClusteringMetrics:
    def __init__(self, features: np.array, labels: np.array) -> None:
        self.features = features
        self.labels = labels
        self.silhouet_index = None
        self.dunn_index = None
        self.davies_bouldin_index = None

    def calculate_silhouet_index(self) -> float:
        if self.silhouet_index is None:
            self.silhouet_index = silhouette_score(self.features, self.labels)

        return self.silhouet_index

    def calculate_dunn_index(self) -> float:
        if self.dunn_index is None:
            self.dunn_index = dunn_fast(self.features, self.labels)

        return self.dunn_index

    def calculate_davies_bouldin_index(self) -> float:
        if self.davies_bouldin_index is None:
            self.davies_bouldin_index = davies_bouldin_score(self.features, self.labels)

        return self.davies_bouldin_index

    def calculate_all_indices(self):
        self.calculate_silhouet_index()
        self.calculate_dunn_index()
        self.calculate_davies_bouldin_index()

