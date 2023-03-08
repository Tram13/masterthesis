import logging

import pandas as pd
from pathlib import Path

from tqdm import tqdm

from src.NLP.main_online_BERTopic import create_model_online_BERTopic, create_scores_from_online_model
from src.NLP.sentence_splitter import SentenceSplitter
from src.data.data_preparer import DataPreparer
from src.data.data_reader import DataReader
from src.tools.config_parser import ConfigParser
from src.user_profile_creation import calculate_basic_user_profiles
from multiprocessing import Pool


def pool_func(param):
    return calculate_profile_by_user_pool(param[0], param[1])


def calculate_profile_by_user_pool(reviews: pd.Series, user_id: str, use_cache=True, save_in_cache=True):
    current_save_dir = Path(ConfigParser().get_value('data', 'user_profiles_path'))
    if not current_save_dir.is_dir():
        current_save_dir.mkdir()

    save_path = current_save_dir.joinpath(f"{user_id}.parquet")

    if use_cache:
        try:
            user_profiles = pd.read_parquet(save_path, engine='fastparquet')
            return user_profiles
        except OSError:
            pass

    scores = create_scores_from_online_model(reviews, use_cache=False, save_in_cache=False, verbose=False)
    user_profiles = scores.aggregate(['mean'], axis=0)
    user_profiles.columns = [str(x) for x in user_profiles.columns]

    if save_in_cache:
        user_profiles.to_parquet(save_path, engine='fastparquet')

    return user_profiles


def calculate_profile_by_user(reviews: pd.DataFrame, user_id: str, use_cache=True, save_in_cache=True):
    current_save_dir = Path(ConfigParser().get_value('data', 'user_profiles_path'))
    if not current_save_dir.is_dir():
        current_save_dir.mkdir()

    save_path = current_save_dir.joinpath(f"{user_id}.parquet")

    if use_cache:
        try:
            user_profiles = pd.read_parquet(save_path, engine='fastparquet')
            return user_profiles
        except OSError:
            pass

    reviews = reviews.loc[reviews['user_id'] == user_id]

    scores = create_scores_from_online_model(reviews['text'], use_cache=False, save_in_cache=False, verbose=False)

    user_profiles = calculate_basic_user_profiles(reviews, scores)
    user_profiles.columns = [str(x) for x in user_profiles.columns]

    if save_in_cache:
        user_profiles.to_parquet(save_path, engine='fastparquet')

    return user_profiles


def main():
    print("hello world")
    _, reviews, _ = DataReader().read_data()

    print('Finished reading in data, starting NLP...')
    # create a fitted online model with the data
    # create_model_online_BERTopic(reviews['text'], model_name="online_bert_big_model.bert")
    # gather the scores based of the current model

    logging.basicConfig(level=logging.WARNING)

    # too slow
    # reviews = [(group['text'], name) for name, group in reviews.groupby('user_id')]

    # for review, user in tqdm(reviews):
    #   calculate_profile_by_user_pool(review, user)

    # to memory heavy code:
    # scores = create_scores_from_online_model(reviews['text'], use_cache=True, save_in_cache=False)
    # print('creating user profiles...')
    # user_profiles = calculate_basic_user_profiles(reviews, scores)
    # user_profiles.columns = [str(x) for x in user_profiles.columns]
    # user_profiles.to_parquet(Path('NLP/TEST_USER_PROFILES.parquet'), engine='fastparquet')


if __name__ == '__main__':
    main()
