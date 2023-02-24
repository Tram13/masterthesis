import pandas as pd

from src.NLP.Models.SBERT_feature_extraction import SentenceBERT
from src.NLP.Models.dimensionality_reduction import DimensionalityReduction
from src.NLP.Models.clustering import ClusteringAlgorithms


# add cluster labels to the DateFrame
def cluster_sentences(reviews: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # get sentence embedding for each sentence of the review
    feature_extractor = SentenceBERT()
    df_features = pd.DataFrame(feature_extractor.get_features(reviews['text']))

    # dimensionality reduction
    dim_reducer = DimensionalityReduction(df_features)
    reduced_features = dim_reducer.features_UMAP()

    # clustering
    clustering_algorithms = ClusteringAlgorithms(reduced_features)
    clustering_labels, amount_clusters = clustering_algorithms.labels_KMEANS()  # todo finetune this, more options

    reviews['cluster_labels'] = clustering_labels
    return reviews, amount_clusters
