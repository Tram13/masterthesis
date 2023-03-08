import pandas as pd


def calculated_basic_profile_for_one_user(reviews: pd.Series, scores: pd.DataFrame):
    pass


# only based on user_id and feature vector of each review
def calculate_basic_user_profiles(reviews: pd.DataFrame, scores: pd.DataFrame):
    scores_with_userid = pd.concat([reviews.reset_index(drop=True)['user_id'], scores], axis=1)
    return scores_with_userid.groupby('user_id').agg('mean').reset_index()


def calculate_time_based_user_profiles(reviews: pd.DataFrame, scores: pd.DataFrame):
    raise NotImplementedError
