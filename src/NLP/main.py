import pandas as pd
import numpy as np

from src.NLP.Models.BERTopic import CustomBERTTopic
from src.NLP.df_NLP_manipulation.df_clustering import cluster_sentences
from src.NLP.df_NLP_manipulation.df_sentiment_analysis import sentiment_analysis_sentences
from src.NLP.scoring_functions import bertopic_scoring_func, basic_clustering_scoring_func
from src.NLP.sentence_splitter import split_reviews


def main_BERTopic(reviews: pd.Series, precompute_and_save_embeddings: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    # split reviews into sentences
    reviews = split_reviews(reviews)

    # create the model
    BERTopic_model = CustomBERTTopic(max_topics=50).model

    embeddings = None
    if precompute_and_save_embeddings:
        pass

    # generate topics
    topics, probabilities = BERTopic_model.fit_transform(reviews['text'], embeddings=embeddings)
    force_topics = BERTopic_model.reduce_outliers(documents=reviews['text'], topics=topics, threshold=0.1)

    # add them to the dataframe
    col_names = list(reviews.columns) + ['topic_id', 'force_topic_id', 'topic_probability']
    reviews = pd.concat([reviews, pd.Series(topics), pd.Series(force_topics), pd.Series(probabilities)], axis=1)
    reviews.columns = col_names

    # sentiment label+score for each sentence
    reviews = sentiment_analysis_sentences(reviews)

    # merge sentences back to one review
    reviews = reviews.groupby('index').aggregate(lambda item: item.tolist())

    # convert elements to numpy array
    reviews[['topic_id', 'force_topic_id', 'topic_probability', 'label_sentiment', 'score_sentiment']] = reviews[
        ['topic_id', 'force_topic_id', 'topic_probability', 'label_sentiment', 'score_sentiment']].applymap(
        np.array)

    # aggregate scores using a custom formula
    bert_scores = reviews[
        ['topic_id', 'force_topic_id', 'topic_probability', 'label_sentiment', 'score_sentiment']].apply(
        bertopic_scoring_func, total_amount_topics=len(BERTopic_model.get_topic_info()['Topic']),
        weight_main_topics=0.75, axis=1)

    return bert_scores, BERTopic_model.get_topic_info()


def main_basic_clustering(reviews: pd.Series):
    # split reviews into sentences
    splitted_reviews = split_reviews(reviews)

    # clustering label
    reviews, amount_clusters = cluster_sentences(splitted_reviews)

    # sentiment label+score for each sentence
    reviews = sentiment_analysis_sentences(reviews)

    # merge sentences back to one review
    reviews = reviews.groupby('index').aggregate(lambda item: item.tolist())

    # convert elements to numpy array
    reviews[['label_sentiment', 'score_sentiment']] = reviews[['label_sentiment', 'score_sentiment']].applymap(np.array)

    # aggregate scores using a custom formula based on cluster and sentiment
    reviews['cluster_scores'] = reviews[['cluster_labels', 'label_sentiment', 'score_sentiment']].apply(
        basic_clustering_scoring_func, total_amount_topics=amount_clusters, axis=1
    )

    return reviews  # todo niet alles moet gereturned worden


if __name__ == '__main__':
    reviews_input = pd.read_csv('tmp.pd')['text']
    # reviews_input_big = pd.read_csv('full_tmp.pd')['text'].head(100_000)
    x = main_basic_clustering(reviews_input)

    print(x)
