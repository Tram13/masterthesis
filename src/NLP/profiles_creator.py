import logging

import pandas as pd

from NLP.main_user_profiles import main_user_profile_topic, main_user_profile_approximation


class ProfileCreator:
    # manual preselection for the 400 topic model. Test if filtering out topics works better
    FILTER_USERS_400TOPS = [
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
    FILTER_BUSINESS_400TOPS = [
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

    def __init__(self, model_name: str,
                 use_sentiment_in_scores: bool,
                 approx_mode: bool,
                 approx_normalization: bool,
                 approx_amount_top_n: int,
                 filter_useful_topics: bool):
        # the model to be used
        self.current_model_name = model_name
        self.batches, self.approx_batches, self.approx_top_n = self._get_amount_of_batches_for_model(model_name)

        # general parameters
        self.use_sentiment_in_scores = use_sentiment_in_scores

        # parameters that only apply for approximation
        self.approx_mode = approx_mode
        self.approx_normalization = approx_normalization    # normalize after selecting top_n
        self.approx_amount_top_n = approx_amount_top_n      # set N for top_n selection in approximation
        self.filter_useful_topics = filter_useful_topics    # filter out certain topics (manual selection)

        # can only be done for this model
        if model_name != "online_model_400top_97.bert" and filter_useful_topics:
            logging.warning("FILTERING TOPICS FOR WRONG MODEL")

    def __str__(self):
        return str(self.get_build_parameters())

    def __repr__(self):
        return self.__str__()

    def get_build_parameters(self) -> dict:
        return {
            "current_model_name": self.current_model_name,
            "use_sentiment_in_scores": self.use_sentiment_in_scores,
            "approx_mode": self.approx_mode,
            "approx_normalization": self.approx_normalization,
            "approx_amount_top_n": self.approx_amount_top_n,
            "filter_useful_topics": self.filter_useful_topics
        }

    @staticmethod
    def load_from_dict(params: dict):
        return ProfileCreator(
            model_name=params['current_model_name'],
            use_sentiment_in_scores=params['use_sentiment_in_scores'],
            approx_mode=params['approx_mode'],
            approx_normalization=params['approx_normalization'],
            approx_amount_top_n=params['approx_amount_top_n'],
            filter_useful_topics=params['filter_useful_topics']
        )

    @staticmethod
    def _get_amount_of_batches_for_model(model_name):
        # hardcoded here
        # score batches, approximation batches, top_n batches
        if model_name == "offline_bertopic_100000.bert":
            # offline bertopic does not have the approximations in cache (yet)
            return 10, None, None
        elif model_name == "online_model_50top_85.bert":
            return 10, 1, 10
        elif model_name == "online_model_400top_97.bert":
            return 80, 8, 80
        elif model_name == "BERTopic_400_dim_red_100.bert":
            return 10, 8, 80

    def get_user_profile(self, reviews: pd.DataFrame):
        if self.approx_mode:
            return self._get_profile_approximation(reviews, "user_id", [str(topic) for topic in self.FILTER_USERS_400TOPS] if self.filter_useful_topics else None)
        else:
            return self._get_profile_topic(reviews, "user_id")

    def get_restaurant_profile(self, reviews: pd.DataFrame):
        if self.approx_mode:
            return self._get_profile_approximation(reviews, "business_id", [str(topic) for topic in self.FILTER_BUSINESS_400TOPS] if self.filter_useful_topics else None)
        else:
            return self._get_profile_topic(reviews, "business_id")

    def _get_profile_topic(self, reviews: pd.DataFrame, profile_mode: str):
        return main_user_profile_topic(reviews=reviews,
                                       amount_of_batches=self.batches,
                                       profile_name=f"profile_tmp.parquet",
                                       use_cache=True,
                                       model_name=self.current_model_name,
                                       use_sentiment_in_scores=self.use_sentiment_in_scores,
                                       profile_mode=profile_mode,
                                       part_of_dataset=True
                                       )

    def _get_profile_approximation(self, reviews: pd.DataFrame, profile_mode: str, filter_select: None):
        return main_user_profile_approximation(reviews=reviews,
                                               amount_of_batches_for_approximations=self.approx_batches,
                                               model_name=self.current_model_name,
                                               amount_of_batches_top_n=self.approx_top_n,
                                               profile_name=f"profile_tmp.parquet",
                                               use_cache=True,
                                               use_splitted_cache=True,
                                               normalize_after_selection=self.approx_normalization,
                                               top_n_topics=self.approx_amount_top_n,
                                               profile_mode=profile_mode,
                                               filter_select=filter_select,
                                               part_of_dataset=True
                                               )
