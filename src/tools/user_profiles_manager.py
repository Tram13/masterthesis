from NLP.profiles_creator import ProfileCreator


class UserProfilesManager:

    USER_PROFILES_NLP = [
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
        # ProfileCreator(  # TODO opnieuw aanzetten
        #     model_name="online_model_400top_97.bert",
        #     use_sentiment_in_scores=False,
        #     approx_mode=True,
        #     approx_normalization=True,
        #     approx_amount_top_n=10,
        #     filter_useful_topics=True
        # ).get_build_parameters(),
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
            use_sentiment_in_scores=False,
            approx_mode=False,
            approx_normalization=True,
            approx_amount_top_n=5,
            filter_useful_topics=False
        ).get_build_parameters(),
    ]

    def __next__(self) -> dict:
        if self.user_index < len(self.USER_PROFILES_NLP):
            config = self.USER_PROFILES_NLP[self.user_index]
            self.user_index += 1
            return config
        else:
            raise StopIteration

    def __len__(self):
        return len(self.USER_PROFILES_NLP)

    def __iter__(self):
        self.user_index = 0
        return self
