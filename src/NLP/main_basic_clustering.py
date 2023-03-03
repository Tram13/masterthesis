import pandas as pd
import numpy as np

from src.NLP.df_NLP_manipulation.df_clustering import cluster_sentences
from src.NLP.df_NLP_manipulation.df_sentiment_analysis import sentiment_analysis_sentences
from src.NLP.scoring_functions import basic_clustering_scoring_func
from src.NLP.sentence_splitter import SentenceSplitter


def main_basic_clustering(reviews: pd.Series):
    # split reviews into sentences
    sentence_splitter = SentenceSplitter()
    splitted_reviews = sentence_splitter.split_reviews(reviews)

    # clustering label
    reviews, amount_clusters = cluster_sentences(splitted_reviews)

    # sentiment label+score for each sentence
    reviews = sentiment_analysis_sentences(reviews)

    # merge sentences back to one review
    reviews = reviews.groupby('review_id').aggregate(lambda item: item.tolist())

    # convert elements to numpy array
    reviews[['label_sentiment', 'score_sentiment']] = reviews[['label_sentiment', 'score_sentiment']].applymap(np.array)

    # aggregate scores using a custom formula based on cluster and sentiment
    reviews['cluster_scores'] = reviews[['cluster_labels', 'label_sentiment', 'score_sentiment']].apply(
        basic_clustering_scoring_func, total_amount_topics=amount_clusters, axis=1
    )

    return reviews  # todo niet alles moet gereturned worden
