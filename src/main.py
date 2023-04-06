import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from NLP.main_online_BERTopic import create_model_online_BERTopic, create_scores_from_online_model_by_topic
from NLP.main_user_profiles import main_user_profile_approximation, main_user_profile_topic
from NLP.managers.nlp_cache_manager import NLPCache
from NLP.managers.nlp_model_manager import NLPModels
from NLP.utils.sentence_splitter import SentenceSplitter
from data.data_reader import DataReader


def main_user_profile_approximation_400topics():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    # reviews = reviews.head(1000)

    logging.info('Finished reading in data, starting NLP...')
    # todo manual filtering of topics
    main_user_profile_approximation(reviews,
                                    amount_of_batches_for_approximations=8,
                                    model_name="online_model_400top_97.bert",
                                    amount_of_batches_top_n=80,
                                    profile_name="APPROX_USER_PROFILES_top_5_400filtered_topics"
                                    )


def main_user_profile_400topics():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()

    logging.info('Finished reading in data, starting NLP...')

    main_user_profile_topic(reviews, amount_of_batches=10, profile_name="BASIC_USER_PROFILES_50_no_sentiment.parquet",
                            use_cache=True, model_name="online_model_50top_85.bert", use_sentiment_in_scores=False)

    logging.info(
        '------------------------------------\n\n\n STARTING SECOND USER PROFILE \n\n\n------------------------------')

    main_user_profile_topic(reviews, amount_of_batches=10,
                            profile_name="BASIC_USER_PROFILES_400_no_sentiment_TEST.parquet",
                            use_cache=True, model_name="online_model_400top_97.bert", use_sentiment_in_scores=False)


def main():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    # reviews = reviews.head(1000)

    logging.info('Finished reading in data, starting NLP...')
    main_user_profile_approximation(reviews)


if __name__ == '__main__':
    main_user_profile_400topics()
