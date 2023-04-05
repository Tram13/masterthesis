import numpy as np
import pandas as pd


def calculated_basic_profile_for_one_user(reviews: pd.Series, scores: pd.DataFrame):
    # too slow (batch is too small)
    pass


# only based on user_id and feature vector of each review
def calculate_basic_user_profiles(reviews: pd.DataFrame, scores: pd.DataFrame, agg_type: str = 'mean'):
    scores_with_userid = pd.concat([reviews.reset_index(drop=True)['user_id'], scores], axis=1)
    return scores_with_userid.groupby('user_id').agg(agg_type).reset_index()


def calculate_time_based_user_profiles(reviews: pd.DataFrame, scores: pd.DataFrame):
    raise NotImplementedError


def select_top_n(row: pd.Series, n: int):
    top_n = np.argpartition(row, -n)[-n:]
    top_n_indices = top_n[np.argsort(-row[top_n])]
    return row.where(row.index.isin([str(x) for x in top_n_indices]), 0)


def normalize_user_profile(row: pd.Series):
    if row.max() < 0.000001:
        return row
    return (row-row.min())/(row.max()-row.min())
