import logging

import numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from libs.jqmcvi_base import dunn_fast


class ClusteringMetrics:
    def __init__(self, features: np.array, labels: np.array) -> None:
        self.features = features
        self.labels = labels
        self.silhouette_index = None          # -1 to 1 => close to 1 is good
        self.dunn_index = None              #
        self.davies_bouldin_index = None    # as high as possible
        self.calinski_harabasz_score = None

    def calculate_silhouette_index(self) -> float:
        if self.silhouette_index is None:
            self.silhouette_index = silhouette_score(self.features, self.labels)

        return self.silhouette_index

    def calculate_dunn_index(self) -> float:
        if self.dunn_index is None:
            self.dunn_index = dunn_fast(self.features, self.labels)

        return self.dunn_index

    def calculate_davies_bouldin_index(self) -> float:
        if self.davies_bouldin_index is None:
            self.davies_bouldin_index = davies_bouldin_score(self.features, self.labels)

        return self.davies_bouldin_index

    def calculate_calinski_harabasz_score(self) -> float:
        if self.calinski_harabasz_score is None:
            self.calinski_harabasz_score = calinski_harabasz_score(self.features, self.labels)

        return self.calinski_harabasz_score

    def calculate_all_indices(self):
        print("calculating silhouette...")
        self.calculate_silhouette_index()
        print("calculating dunn...")
        self.calculate_dunn_index()
        print("calculating davies_bouldin...")
        self.calculate_davies_bouldin_index()
        print("calinski_harabasz_score...")
        self.calinski_harabasz_score()

    def __str__(self):
        return f"""Scores:
        Silhouette_index: {self.silhouette_index}
        Dunn_index: {self.dunn_index}
        Davies_bouldin_index: {self.davies_bouldin_index}
        Calinski_harabasz_score: {self.calinski_harabasz_score}
        """

