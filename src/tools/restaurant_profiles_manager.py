from NLP.profiles_creator import ProfileCreator


class RestaurantProfilesManager:

    RESTAURANT_PROFILES_NLP = [
        ProfileCreator(
            model_name="online_model_400top_97.bert",
            use_sentiment_in_scores=False,
            approx_mode=False,
            approx_normalization=True,
            approx_amount_top_n=5,
            filter_useful_topics=False
        ).get_build_parameters(),
        ProfileCreator(
            model_name="online_model_50top_85.bert",
            use_sentiment_in_scores=False,
            approx_mode=False,
            approx_normalization=True,
            approx_amount_top_n=5,
            filter_useful_topics=False
        ).get_build_parameters(),
        ProfileCreator(
            model_name="online_model_50top_85.bert",
            use_sentiment_in_scores=False,
            approx_mode=True,
            approx_normalization=True,
            approx_amount_top_n=5,
            filter_useful_topics=False
        ).get_build_parameters(),
        ProfileCreator(
            model_name="online_model_400top_97.bert",
            use_sentiment_in_scores=True,
            approx_mode=False,
            approx_normalization=True,
            approx_amount_top_n=5,
            filter_useful_topics=False
        ).get_build_parameters(),
        ProfileCreator(
            model_name="online_model_50top_85.bert",
            use_sentiment_in_scores=True,
            approx_mode=False,
            approx_normalization=True,
            approx_amount_top_n=5,
            filter_useful_topics=False
        ).get_build_parameters(),
        ProfileCreator(
            model_name="BERTopic_guided_maxtop_58.bert",
            use_sentiment_in_scores=True,
            approx_mode=False,
            approx_normalization=True,
            approx_amount_top_n=5,
            filter_useful_topics=False
        ).get_build_parameters(),
        # ProfileCreator(
        #     model_name="offline_bertopic_100000.bert",
        #     use_sentiment_in_scores=True,
        #     approx_mode=False,
        #     approx_normalization=True,
        #     approx_amount_top_n=5,
        #     filter_useful_topics=False
        # ).get_build_parameters(),
        # ProfileCreator(
        #     model_name="online_model_400top_97.bert",
        #     use_sentiment_in_scores=True,
        #     approx_mode=True,
        #     approx_normalization=True,
        #     approx_amount_top_n=10,
        #     filter_useful_topics=True
        # ).get_build_parameters(),
    ]

    def get_best(self) -> dict:
        return self.RESTAURANT_PROFILES_NLP[3]

    def __next__(self) -> dict:
        if self.restaurant_index < len(self.RESTAURANT_PROFILES_NLP):
            config = self.RESTAURANT_PROFILES_NLP[self.restaurant_index]
            self.restaurant_index += 1
            return config
        else:
            raise StopIteration

    def __len__(self):
        return len(self.RESTAURANT_PROFILES_NLP)

    def __iter__(self):
        self.restaurant_index = 0
        return self

    def __getitem__(self, i):
        return self.RESTAURANT_PROFILES_NLP[i]
