import numpy as np
import pandas as pd


# only based on user_id and feature vector of each review
def calculate_basic_user_profiles_from_series(reviews: pd.DataFrame, scores: pd.Series, agg_type: str = 'mean', mode: str = 'user_id'):
    return reviews[[mode]].join(scores.rename("bert_scores"), on='review_id', how='inner').groupby(mode).agg(agg_type)


def calculate_basic_user_profiles_from_dataframe(reviews: pd.DataFrame, scores: pd.DataFrame, agg_type: str = 'mean', mode: str = 'user_id'):
    return reviews[[mode]].join(scores, on='review_id', how='inner').groupby(mode).agg(agg_type)


def select_top_n(row: pd.Series, n: int):
    top_n = np.argpartition(row, -n)[-n:]
    top_n_indices = top_n[np.argsort(-row[top_n])]
    return row.where(row.index.isin([str(x) for x in top_n_indices]), 0)


def normalize_user_profile(row: pd.Series):
    if row.max() < 0.000001:
        return row
    return (row-row.min())/(row.max()-row.min())
