import pandas as pd
import numpy as np

from pathlib import Path
from src.NLP.ClusteringMetrics import ClusteringMetrics
from src.NLP.ModelsImplementations.CustomBERTopic import CustomBERTTopic
from src.NLP.df_NLP_manipulation.df_sentiment_analysis import sentiment_analysis_sentences
from src.NLP.scoring_functions import bertopic_scoring_func
from src.NLP.sentence_splitter import SentenceSplitter


def main_BERTopic(reviews: pd.Series, embeddings: np.ndarray = None, do_precompute_and_save_embeddings: bool = False,
                  save_path: Path = None) -> tuple[pd.Series, pd.DataFrame]:
    # split reviews into sentences
    print('Splitting Sentences...')
    sentence_splitter = SentenceSplitter()
    reviews = sentence_splitter.split_reviews(reviews)
    input_data = reviews['text']

    # create the model
    BERTopic = CustomBERTTopic(max_topics=50, batch_size=32)
    BERTopic_model = BERTopic.model

    print('Creating Embeddings...')
    # if there are no given embeddings, it is possible to precompute them and save them
    if embeddings is None and do_precompute_and_save_embeddings:
        embeddings = BERTopic.precompute_and_save_embeddings(input_data, save_path=save_path)

    # generate topics
    topics, probabilities = BERTopic_model.fit_transform(input_data, embeddings=embeddings)
    force_topics = BERTopic_model.reduce_outliers(documents=input_data, topics=topics, threshold=0.1)

    print("finished BERTopic, calculating metrics")

    # clustering Metrics
    clustering_metric_default = ClusteringMetrics(embeddings, topics)
    clustering_metric_force_topic = ClusteringMetrics(embeddings, force_topics)

    clustering_metric_default.calculate_all_indices()
    clustering_metric_force_topic.calculate_all_indices()

    print(f'Metrics for default: {str(clustering_metric_default)}\n')
    print(f'Metrics for forced: {str(clustering_metric_force_topic)}\n')

    # add them to the dataframe
    col_names = list(reviews.columns) + ['topic_id', 'force_topic_id', 'topic_probability']
    reviews = pd.concat([reviews, pd.Series(topics), pd.Series(force_topics), pd.Series(probabilities)], axis=1)
    reviews.columns = col_names

    print('calculating sentiment...')
    # sentiment label+score for each sentence
    reviews = sentiment_analysis_sentences(reviews)

    # merge sentences back to one review
    reviews = reviews.groupby('review_id').aggregate(lambda item: item.tolist())

    # convert elements to numpy array
    reviews[['topic_id', 'force_topic_id', 'topic_probability', 'label_sentiment', 'score_sentiment']] = reviews[
        ['topic_id', 'force_topic_id', 'topic_probability', 'label_sentiment', 'score_sentiment']].applymap(
        np.array)

    print('calculating final scores...')

    # aggregate scores using a custom formula
    bert_scores = reviews[
        ['topic_id', 'force_topic_id', 'topic_probability', 'label_sentiment', 'score_sentiment']].apply(
        bertopic_scoring_func, total_amount_topics=len(BERTopic_model.get_topic_info()['Topic']),
        weight_main_topics=0.75, axis=1)

    return bert_scores, BERTopic_model.get_topic_info()
