import logging

import numpy as np
import pandas as pd
from pathlib import Path

from bertopic import BERTopic
from tqdm import tqdm

from src.NLP.df_NLP_manipulation.df_zero_shot_class import zero_shot_class
from src.NLP.main_online_BERTopic import create_scores_from_online_model
from src.NLP.scoring_functions import online_bertopic_scoring_func
from src.NLP.sentence_splitter import SentenceSplitter
from src.data.data_preparer import DataPreparer
from src.data.data_reader import DataReader
from src.tools.config_parser import ConfigParser
from src.user_profile_creation import calculate_basic_user_profiles


def main_user_profile():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    reviews = reviews.head(1000)

    logging.info('Finished reading in data, starting NLP...')
    # create a fitted online model with the data
    # create_model_online_BERTopic(reviews['text'], model_name="online_bert_big_model.bert")
    # gather the scores based of the current model

    # too slow
    # reviews = [(group['text'], name) for name, group in reviews.groupby('user_id')]

    # for review, user in tqdm(reviews):
    #   calculate_profile_by_user_pool(review, user)

    cache_path = Path(ConfigParser().get_value('data', 'nlp_cache_dir'))
    if not cache_path.is_dir():
        cache_path.mkdir()

    amount_of_batches = 10
    for index, batch in enumerate(tqdm(np.array_split(reviews, amount_of_batches), desc="Score Batches")):
        print()
        scores = create_scores_from_online_model(batch['text'], use_cache=False, save_in_cache=False, early_return=True)
        scores.columns = [str(x) for x in scores.columns]
        scores.to_parquet(Path(cache_path, f"score_part_{index}.parquet"), engine='fastparquet')

    scores = pd.read_parquet(Path(cache_path, f"score_part_{0}.parquet"), engine='fastparquet')
    for index in range(1, amount_of_batches):
        to_add = pd.read_parquet(Path(cache_path, f"score_part_{index}.parquet"), engine='fastparquet')
        scores = pd.concat([scores, to_add])

    # merge sentences back to one review
    logging.info('Merging Reviews...')

    scores = scores.groupby('review_id').aggregate(lambda item: item.tolist())
    # convert elements to numpy array
    scores[['topic_id', 'label_sentiment', 'score_sentiment']] = scores[
        ['topic_id', 'label_sentiment', 'score_sentiment']].applymap(
        np.array)

    logging.info("Loading in model...")

    current_save_dir = Path(ConfigParser().get_value('data', 'online_bert_model_path'))
    if not current_save_dir.is_dir():
        current_save_dir.mkdir()
    current_model_save_path = current_save_dir.joinpath(
        Path(ConfigParser().get_value('data', 'use_bert_model_fname')))
    model_online_BERTopic: BERTopic = BERTopic.load(current_model_save_path)

    logging.info("Calculating scores...")
    logging.info("Calculating scores...")
    bert_scores = scores[
        ['topic_id', 'label_sentiment', 'score_sentiment']].apply(
        online_bertopic_scoring_func, total_amount_topics=len(model_online_BERTopic.get_topic_info()['Topic']), axis=1)

    bert_scores = pd.DataFrame(bert_scores.to_list())

    logging.info('creating user profiles...')
    user_profiles = calculate_basic_user_profiles(reviews, bert_scores)
    user_profiles.columns = [str(x) for x in user_profiles.columns]

    logging.info('Saving user profiles...')
    user_profiles.to_parquet(Path('NLP/FIRST_USER_PROFILES.parquet'), engine='fastparquet')


def main_zero_shot_classification():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    reviews = reviews.head(1000)

    logging.info('Finished reading in data, starting NLP...')
    cache_path = Path(ConfigParser().get_value('data', 'nlp_cache_dir'))
    if not cache_path.is_dir():
        cache_path.mkdir()

    classes = ["food", "service", "environment"]

    amount_of_batches = 10
    for index, batch in enumerate(tqdm(np.array_split(reviews, amount_of_batches), desc="Score Batches")):
        print()
        zero_shot_features = zero_shot_class(batch['text'], classes=classes)
        zero_shot_features.columns = [str(x) for x in zero_shot_features.columns]
        zero_shot_features.to_parquet(Path(cache_path, f"zero_shot_classes_{index}.parquet"), engine='fastparquet')

    zero_shot_features = pd.read_parquet(Path(cache_path, f"zero_shot_classes_{0}.parquet"), engine='fastparquet')
    for index in range(1, amount_of_batches):
        to_add = pd.read_parquet(Path(cache_path, f"zero_shot_classes_{index}.parquet"), engine='fastparquet')
        zero_shot_features = pd.concat([zero_shot_features, to_add])

    return zero_shot_features


if __name__ == '__main__':
    main_zero_shot_classification()
