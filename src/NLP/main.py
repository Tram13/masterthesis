import umap
from sklearn.cluster import KMeans
import pandas as pd
from src.NLP.Models.SBERT_feature_extraction import SentenceBERT
from src.NLP.dimensionality_reduction import DimensionalityReduction
from src.NLP.sentence_splitter import SentenceSplitter
import matplotlib.pyplot as plt


def run_main(reviews: list[str]):
    # split reviews into sentences
    splitter = SentenceSplitter()
    splitted_reviews = [splitter.split_text_into_sentences(review) for review in reviews]

    # get sentence embedding for each sentence of the review
    feature_extractor = SentenceBERT()
    extracted_features = [feature_extractor.get_features(review_sentences) for review_sentences in splitted_reviews]

    # reduce the dimension of the extracted features before clustering
    dim_reducer = DimensionalityReduction(extracted_features)
    reduced_features = dim_reducer.features_UMAP()

    # clustering
    kmeans = KMeans(
        init="random",
        n_clusters=3,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    kmeans.fit(reduced_features)

    # Prepare data
    data_2d = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(reduced_features)
    result = pd.DataFrame(data_2d, columns=['x', 'y'])
    result['labels'] = kmeans.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    run_main([])
