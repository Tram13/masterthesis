import pandas as pd
import numpy as np

from pathlib import Path

from NLP.managers.nlp_cache_manager import NLPCache
from NLP.managers.nlp_model_manager import NLPModels
from NLP.utils.ClusteringMetrics import ClusteringMetrics
from NLP.ModelsImplementations.CustomBERTopic import CustomBERTTopic
from NLP.df_NLP_manipulation.df_sentiment_analysis import sentiment_analysis_sentences
from NLP.utils.scoring_functions import bertopic_scoring_func
from NLP.utils.sentence_splitter import SentenceSplitter


def main_BERTopic(reviews: pd.Series, embeddings: np.ndarray = None, do_precompute_and_save_embeddings: bool = False,
                  save_path: Path = None, use_sentiment: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    # split reviews into sentences
    print('Splitting Sentences...')
    sentence_splitter = SentenceSplitter()
    reviews = sentence_splitter.split_reviews(reviews)
    input_data = reviews['text']

    # create the model
    BERTopic = CustomBERTTopic(max_topics=50, batch_size=32)
    BERTopic_model = BERTopic.model
    BERTopic_model.calculate_probabilities = False
    BERTopic_model.low_memory = True

    print('Creating Embeddings...')
    # if there are no given embeddings, it is possible to precompute them and save them
    if embeddings is None and do_precompute_and_save_embeddings:
        embeddings = BERTopic.precompute_and_save_embeddings(input_data, save_path=save_path)

    # generate topics
    topics, _ = BERTopic_model.fit_transform(input_data, embeddings=embeddings)

    # print("finished BERTopic, calculating metrics")

    # clustering Metrics
    # clustering_metric_default = ClusteringMetrics(embeddings, topics)
    # clustering_metric_default.calculate_all_indices()
    # print(f'Metrics for default: {str(clustering_metric_default)}\n')

    print('SAVING model...')
    nlp_models = NLPModels()
    nlp_models.save_model(BERTopic_model, "offline_test_model.bert")

    print('SAVING TOPICS...')
    scores = pd.concat([pd.Series(topics)], axis=1)
    scores.to_parquet(Path("NLP", "cache", f"score_part_OFFLINE_TEST.parquet"), engine='fastparquet')
    print('DONE saving')
    # add them to the dataframe
    reviews = pd.concat([reviews, scores], axis=1)

    if not use_sentiment:
        print("calculating scores")
        reviews = reviews.groupby('review_id').aggregate(lambda item: item.tolist())

        reviews[['topic_id']] = reviews[['topic_id']].applymap(np.array)

        bert_scores = reviews[
            ['topic_id']].apply(
            bertopic_scoring_func, total_amount_topics=len(BERTopic_model.get_topic_info()['Topic']),
            sentiment=False, axis=1)

        bert_scores.to_parquet(Path("NLP", "cache", f"BERT_scores_FINAL_OFFLINE_TEST.parquet"), engine='fastparquet')
    else:
        print('calculating sentiment...')
        # sentiment label+score for each sentence
        reviews = sentiment_analysis_sentences(reviews)

        # merge sentences back to one review
        reviews = reviews.groupby('review_id').aggregate(lambda item: item.tolist())

        # convert elements to numpy array
        reviews[['topic_id', 'label_sentiment', 'score_sentiment']] = reviews[
            ['topic_id', 'label_sentiment', 'score_sentiment']].applymap(
            np.array)

        print('calculating final scores...')

        # aggregate scores using a custom formula
        bert_scores = reviews[
            ['topic_id', 'label_sentiment', 'score_sentiment']].apply(
            bertopic_scoring_func, total_amount_topics=len(BERTopic_model.get_topic_info()['Topic']), axis=1)

        return bert_scores, BERTopic_model.get_topic_info()
