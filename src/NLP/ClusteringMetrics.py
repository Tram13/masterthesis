import numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.libs.jqmcvi_base import dunn_fast


class ClusteringMetrics:
    def __init__(self, features: np.array, labels: np.array) -> None:
        self.features = features
        self.labels = labels
        self.silhouet_index = None          # -1 to 1 => close to 1 is good
        self.dunn_index = None              #
        self.davies_bouldin_index = None    # as high as possible

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
        print("calculating silhouet...")
        self.calculate_silhouet_index()
        print("calculating dunn...")
        self.calculate_dunn_index()
        print("calculating davies_bouldin...")
        self.calculate_davies_bouldin_index()

    def __str__(self):
        return f"""Scores:
        Silhouet_index: {self.silhouet_index}
        Dunn_index: {self.dunn_index}
        Davies_bouldin_index: {self.davies_bouldin_index}
        """

