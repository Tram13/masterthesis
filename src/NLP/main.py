import pandas as pd

from src.NLP.Models.BasicSentimentAnalysis import BasicSentimentAnalysis
from src.NLP.Models.SBERT_feature_extraction import SentenceBERT
from src.NLP.dimensionality_reduction import DimensionalityReduction
from src.NLP.sentence_splitter import SentenceSplitter
from src.NLP.clustering import ClusteringAlgorithms


# add cluster labels to the DateFrame
def cluster_sentences(reviews: pd.DataFrame) -> pd.DataFrame:
    # get sentence embedding for each sentence of the review
    feature_extractor = SentenceBERT()
    df_features = pd.DataFrame(feature_extractor.get_features(reviews['text']))

    # dimensionality reduction
    dim_reducer = DimensionalityReduction(df_features)
    reduced_features = dim_reducer.features_UMAP()

    # clustering
    clustering_algorithms = ClusteringAlgorithms(reduced_features)
    clustering_labels = clustering_algorithms.labels_KMEANS()   # todo finetune this

    reviews['cluster_labels'] = clustering_labels
    return reviews


# add sentiment label and score to the DataFrame
def sentiment_analysis_sentences(reviews: pd.DataFrame):
    sentiment_analyzer = BasicSentimentAnalysis()
    df_sentiment = sentiment_analyzer.get_sentiment(list(reviews['text']))
    df_sentiment.columns = ['label_sentiment', 'score_sentiment']
    return pd.concat([reviews, df_sentiment], axis=1)


def run_main(reviews: pd.Series):
    splitter = SentenceSplitter()
    splitted_reviews = pd.DataFrame(reviews.map(splitter.split_text_into_sentences))
    # split sentences out in pd.dataframe while keeping indices of review
    splitted_reviews = splitted_reviews.explode('text').reset_index()
    splitted_reviews['text'] = splitted_reviews['text'].map(str.strip)

    # clustering label
    reviews = cluster_sentences(splitted_reviews)

    # sentiment label+score for each sentence
    reviews = sentiment_analysis_sentences(reviews)


if __name__ == '__main__':
    reviews_input = pd.read_csv('tmp.pd')['text']
    x = run_main(reviews_input)
    print(x)

