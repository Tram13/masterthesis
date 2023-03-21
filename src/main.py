import logging

import numpy as np
import pandas as pd
from pathlib import Path

from bertopic import BERTopic
from tqdm import tqdm

from src.NLP.df_NLP_manipulation.df_zero_shot_class import zero_shot_class
from src.NLP.main_online_BERTopic import create_scores_from_online_model
from src.NLP.nlp_cache_manager import NLPCache
from src.NLP.nlp_model_manager import NLPModels
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
    reviews = reviews.head(100)

    logging.info('Finished reading in data, starting NLP...')
    nlp_cache = NLPCache()

    amount_of_batches = 10
    for index, batch in enumerate(tqdm(np.array_split(reviews, amount_of_batches), desc="Score Batches")):
        print()
        scores = create_scores_from_online_model(batch['text'], use_cache=False, save_in_cache=False, early_return=True)
        scores.columns = [str(x) for x in scores.columns]
        scores.to_parquet(nlp_cache.bert_scores_path.joinpath(Path(f"score_part_{index}.parquet"), engine='fastparquet'))

    scores = nlp_cache.load_bert_scores()

    # merge sentences back to one review
    logging.info('Merging Reviews...')

    scores = scores.groupby('review_id').aggregate(lambda item: item.tolist())
    # convert elements to numpy array
    scores[['topic_id', 'label_sentiment', 'score_sentiment']] = scores[
        ['topic_id', 'label_sentiment', 'score_sentiment']].applymap(
        np.array)

    logging.info("Loading in model...")
    model_manager = NLPModels()
    model_online_BERTopic: BERTopic = model_manager.load_model()

    logging.info("Calculating scores...")
    bert_scores = scores[
        ['topic_id', 'label_sentiment', 'score_sentiment']].apply(
        online_bertopic_scoring_func, total_amount_topics=len(model_online_BERTopic.get_topic_info()['Topic']), axis=1)

    bert_scores = pd.DataFrame(bert_scores.to_list())

    logging.info('creating user profiles...')
    user_profiles = calculate_basic_user_profiles(reviews, bert_scores)
    user_profiles.columns = [str(x) for x in user_profiles.columns]

    logging.info('Saving user profiles...')
    user_profiles.to_parquet(nlp_cache.user_profiles_path.joinpath(Path('BASIC_USER_PROFILES.parquet'), engine='fastparquet'))


def main_zero_shot_classification():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    reviews = reviews.head(1000)

    logging.info('Finished reading in data, starting NLP...')
    nlp_cache = NLPCache()

    classes = ["food", "service", "environment"]

    amount_of_batches = 30
    for index, batch in enumerate(tqdm(np.array_split(reviews, amount_of_batches), desc="Score Batches")):
        print()
        zero_shot_features = zero_shot_class(batch['text'], classes=classes)
        zero_shot_features.columns = [str(x) for x in zero_shot_features.columns]
        zero_shot_features.to_parquet(nlp_cache.zero_shot_classes_path.joinpath(Path(f"zero_shot_classes_{index}.parquet"), engine='fastparquet'))

    return nlp_cache.load_zero_shot_classes()


if __name__ == '__main__':
    main_user_profile()
