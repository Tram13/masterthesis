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


def main_user_profile_approximation_400topics(normalize: bool = False):
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    # reviews = reviews.head(1000)

    logging.info('Finished reading in data, starting NLP...')
    # todo manual filtering of topics
    useful_topics_users_400tops_model = [
        1,
        2,
        4,
        9,
        11,
        12,
        14,
        15,
        17,
        18,
        23,
        26,
        28,
        29,
        30,
        34,
        35,
        36,
        37,
        42,
        43,
        44,
        45,
        51,
        59,
        60,
        62,
        64,
        66,
        67,
        70,
        72,
        75,
        79,
        81,
        82,
        89,
        90,
        91,
        95,
        96,
        97,
        99,
        103,
        104,
        105,
        107,
        110,
        112,
        115,
        128,
        130,
        132,
        133,
        134,
        135,
        137,
        143,
        145,
        146,
        148,
        154,
        155,
        162,
        165,
        166,
        176,
        177,
        179,
        180,
        187,
        191,
        192,
        193,
        200,
        207,
        209,
        210,
        212,
        213,
        217,
        218,
        228,
        235,
        237,
        239,
        242,
        246,
        248,
        250,
        257,
        263,
        264,
        272,
        273,
        274,
        275,
        284,
        285,
        286,
        288,
        305,
        306,
        307,
        310,
        311,
        323,
        324,
        325,
        326,
        327,
        330,
        335,
        338,
        341,
        342,
        345,
        346,
        347,
        355,
        356,
        357,
        365,
        366,
        376,
        384,
        394,
        398
    ]
    nlp_models = NLPModels()
    model_name = "online_model_400top_97.bert"
    main_user_profile_approximation(reviews,
                                    amount_of_batches_for_approximations=8,
                                    model_name=model_name,
                                    amount_of_batches_top_n=80,
                                    profile_name=f"APPROX_USER_PROFILES_top_5_400filtered_topics_normalized_{normalize}",
                                    filter_select=useful_topics_users_400tops_model,
                                    approx_save_dir=nlp_models.get_dir_for_model(model_name),
                                    normalize_after_selection=normalize
                                    )


def main_user_profile_approximation_50topics(normalize: bool = False, top_n: int = 5):
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()

    logging.info('Finished reading in data, starting NLP...')
    main_user_profile_approximation(reviews,
                                    amount_of_batches_for_approximations=1,
                                    model_name="online_model_50top_85.bert",
                                    amount_of_batches_top_n=10,
                                    profile_name=None,
                                    normalize_after_selection=normalize,
                                    top_n_topics=top_n
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
                            profile_name="BASIC_USER_PROFILES_400_no_sentiment.parquet",
                            use_cache=True, model_name="online_model_400top_97.bert", use_sentiment_in_scores=False)


def main():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    # reviews = reviews.head(1000)

    logging.info('Finished reading in data, starting NLP...')
    main_user_profile_approximation(reviews)


if __name__ == '__main__':
    logging.info(
        '------------------------------------\n\n\n ALGO 1: approx 50 with normalization \n\n\n------------------------------')
    main_user_profile_approximation_50topics(True, 5)
    logging.info(
        '------------------------------------\n\n\n ALGO 2: approx 400 tops with preselection, no normalization \n\n\n------------------------------')
    main_user_profile_approximation_400topics(False)
    logging.info(
        '------------------------------------\n\n\n ALGO 3: approx 50 tops no normalization top 10 topics\n\n\n------------------------------')
    main_user_profile_approximation_50topics(False, 10)
    logging.info(
        '------------------------------------\n\n\n ALGO 4: approx 50 tops with normalization top 3 topics\n\n\n------------------------------')
    main_user_profile_approximation_50topics(True, 3)
    logging.info(
        '------------------------------------\n\n\n ALGO 5: approx 400 tops with preselection, with extra normalization \n\n\n------------------------------')
    main_user_profile_approximation_400topics(True)
