import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from NLP.main_offline_BERTopic import main_BERTopic
from NLP.main_online_BERTopic import create_model_online_BERTopic, create_scores_from_online_model_by_topic
from NLP.main_user_profiles import main_user_profile_approximation, main_user_profile_topic
from NLP.managers.nlp_cache_manager import NLPCache
from NLP.managers.nlp_model_manager import NLPModels
from NLP.utils.evaluate_model import evaluate_model
from NLP.utils.sentence_splitter import SentenceSplitter
from data.data_reader import DataReader


def main_user_profile_approximation_400topics(normalize: bool = False, top_n: int = 5, profile_mode: str = "user_id"):
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
    useful_topics_business_400tops_model = [
        0,
        1,
        2,
        3,
        4,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        22,
        23,
        25,
        26,
        28,
        29,
        30,
        33,
        34,
        36,
        38,
        42,
        43,
        44,
        45,
        46,
        51,
        52,
        54,
        56,
        59,
        60,
        61,
        62,
        63,
        64,
        66,
        72,
        75,
        78,
        79,
        80,
        81,
        83,
        85,
        86,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        105,
        106,
        108,
        109,
        110,
        112,
        115,
        116,
        118,
        120,
        121,
        125,
        128,
        130,
        135,
        138,
        142,
        143,
        144,
        145,
        146,
        148,
        154,
        155,
        158,
        162,
        164,
        165,
        166,
        170,
        176,
        177,
        182,
        185,
        186,
        187,
        191,
        192,
        193,
        198,
        199,
        201,
        204,
        206,
        207,
        209,
        210,
        211,
        212,
        213,
        222,
        226,
        228,
        231,
        234,
        235,
        236,
        242,
        243,
        246,
        249,
        250,
        253,
        257,
        258,
        259,
        264,
        265,
        266,
        267,
        271,
        273,
        274,
        275,
        276,
        277,
        278,
        284,
        285,
        286,
        288,
        291,
        296,
        298,
        304,
        305,
        306,
        307,
        310,
        311,
        314,
        323,
        325,
        326,
        327,
        330,
        332,
        333,
        335,
        342,
        345,
        347,
        350,
        355,
        356,
        362,
        365,
        366,
        369,
        381,
        382,
        383,
        384,
        394,
        396,
        398,
        399
    ]

    preselect = useful_topics_users_400tops_model if profile_mode == "user_id" else useful_topics_business_400tops_model
    preselect = [str(topic) for topic in preselect]
    nlp_models = NLPModels()
    model_name = "online_model_400top_97.bert"
    main_user_profile_approximation(reviews,
                                    amount_of_batches_for_approximations=8,
                                    model_name=model_name,
                                    top_n_topics=top_n,
                                    amount_of_batches_top_n=80,
                                    profile_name=f"APPROX_{'USER' if profile_mode == 'user_id' else 'BUSINESS'}_PROFILE_top_{top_n}_400filtered_topics_normalized_{normalize}",
                                    filter_select=preselect,
                                    approx_save_dir=nlp_models.get_dir_for_model(model_name),
                                    normalize_after_selection=normalize,
                                    profile_mode=profile_mode
                                    )


def main_user_profile_approximation_50topics(normalize: bool = False, top_n: int = 5, profile_mode: str = "user_id",
                                             profile_name: str = None):
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()

    logging.info('Finished reading in data, starting NLP...')
    main_user_profile_approximation(reviews,
                                    amount_of_batches_for_approximations=1,
                                    model_name="online_model_50top_85.bert",
                                    amount_of_batches_top_n=10,
                                    profile_name=profile_name,
                                    normalize_after_selection=normalize,
                                    top_n_topics=top_n,
                                    profile_mode=profile_mode
                                    )


def main_business_profile_50topics(sentiment: bool = True):
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()

    logging.info('Finished reading in data, starting NLP...')

    main_user_profile_topic(reviews, amount_of_batches=10,
                            profile_name=f"BUSINESS_PROFILE_50_sentiment={sentiment}.parquet",
                            use_cache=True, model_name="online_model_50top_85.bert", use_sentiment_in_scores=True,
                            profile_mode="business_id")


def main_bert_guided():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    nlp_cache = NLPCache()
    topics = nlp_cache.read_guided_topics()
    max_top = 10 + len(topics)

    logging.info('Finished reading in data, starting NLP...')
    create_model_online_BERTopic(reviews['text'], model_name=f"BERTopic_guided_maxtop_{max_top}", max_topics=max_top,
                                 guided_topics=topics)


def main_evaluate_model(model_name):
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    sentences = SentenceSplitter()._load_splitted_reviews_from_cache()

    logging.info('Finished reading in data, starting evaluation...')
    evaluate_model(sentences, model_name)


def main():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    # reviews = reviews.head(1000)

    logging.info('Finished reading in data, starting NLP...')
    main_BERTopic(reviews['text'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('Clustering metric for 1 model')
    main_evaluate_model("online_model_50top_85.bert")
