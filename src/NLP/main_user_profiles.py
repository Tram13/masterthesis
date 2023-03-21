import logging

import numpy as np
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm

from src.NLP.main_online_BERTopic import create_scores_from_online_model
from src.NLP.managers.nlp_cache_manager import NLPCache
from src.NLP.managers.nlp_model_manager import NLPModels
from src.NLP.utils.scoring_functions import online_bertopic_scoring_func
from src.NLP.utils.user_profile_creation import calculate_basic_user_profiles


def main_user_profile(reviews: pd.DataFrame, amount_of_batches: int = 10,
                      profile_name: str = "BASIC_USER_PROFILES.parquet", use_cache: bool = True):
    logging.info('Finished reading in data, starting NLP...')
    nlp_cache = NLPCache()

    if not use_cache or not nlp_cache.is_available_scores():
        logging.warning(f'Cache is not being used: allowed: {use_cache} - available: {nlp_cache.is_available_scores()}')
        logging.info('Calculating bert_scores...')
        for index, batch in enumerate(tqdm(np.array_split(reviews, amount_of_batches), desc="Score Batches")):
            print()
            scores = create_scores_from_online_model(batch['text'], use_cache=False, save_in_cache=False,
                                                     early_return=True)
            scores.columns = [str(x) for x in scores.columns]
            scores.to_parquet(nlp_cache.scores_path.joinpath(Path(f"score_part_{index}.parquet"), engine='fastparquet'))

    logging.info('Loading in all scores...')
    scores = nlp_cache.load_scores()

    # merge sentences back to one review
    logging.info('Merging Reviews...')

    scores = scores.groupby('review_id').aggregate(lambda item: item.tolist())
    # convert elements to numpy array
    scores[['topic_id', 'label_sentiment', 'score_sentiment']] = scores[
        ['topic_id', 'label_sentiment', 'score_sentiment']].applymap(
        np.array)

    logging.info("Loading in model...")
    model_manager = NLPModels()
    model_online_BERTopic: BERTopic = model_manager.load_model()

    logging.info("Calculating bert_scores...")
    bert_scores = scores[
        ['topic_id', 'label_sentiment', 'score_sentiment']].apply(
        online_bertopic_scoring_func, total_amount_topics=len(model_online_BERTopic.get_topic_info()['Topic']), axis=1)

    bert_scores = pd.DataFrame(bert_scores.to_list())

    logging.info('creating user profiles from bert_scores...')
    user_profiles = calculate_basic_user_profiles(reviews, bert_scores)
    user_profiles.columns = [str(x) for x in user_profiles.columns]

    logging.info('Saving user profiles...')
    user_profiles.to_parquet(nlp_cache.user_profiles_path.joinpath(Path(profile_name), engine='fastparquet'))
