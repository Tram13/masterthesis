import logging

import numpy as np
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm

from NLP.main_online_BERTopic import create_scores_from_online_model_by_topic, create_scores_by_approximate_distribution
from NLP.managers.nlp_cache_manager import NLPCache
from NLP.managers.nlp_model_manager import NLPModels
from NLP.utils.scoring_functions import online_bertopic_scoring_func
from NLP.utils.sentence_splitter import SentenceSplitter
from NLP.utils.user_profile_creation import calculate_basic_user_profiles, select_top_n, normalize_user_profile


def main_user_profile_approximation(reviews: pd.DataFrame, amount_of_batches_for_approximations: int = 1,
                                    model_name: str = None,
                                    profile_name: str = None, use_cache: bool = True,
                                    use_splitted_cache: bool = True, top_n_topics: int = 5,
                                    approx_save_dir: str = "base", prefilter_select: list[int] = None):
    if profile_name is None:
        profile_name = f"APPROX_USER_PROFILES_top_{top_n_topics}.parquet"

    logging.info('Finished reading in data, starting NLP...')
    nlp_cache = NLPCache()

    if not use_cache or not nlp_cache.is_available_approximation(approx_save_dir):
        logging.warning(
            f'Cache is not being used for approximations: allowed: {use_cache} - available: {nlp_cache.is_available_approximation(approx_save_dir)}')
        logging.info('Calculating approximations for current model...')
        for index, batch in enumerate(
                tqdm(np.array_split(reviews, amount_of_batches_for_approximations), desc="Approximation batches")):
            print()
            approximation = create_scores_by_approximate_distribution(batch['text'], model_name=model_name,
                                                                      use_cache=use_splitted_cache)
            approximation.columns = [str(x) for x in approximation.columns]
            approximation.to_parquet(
                nlp_cache.approximation_path.joinpath(Path(approx_save_dir, f"approximation_part_{index}.parquet")),
                engine='fastparquet')

    logging.info('Loading in approximations...')
    topic_distributions = nlp_cache.load_approximation(approx_save_dir)

    # select only topics that are relevant and not too general
    if prefilter_select:
        topic_distributions = topic_distributions[prefilter_select]

    logging.info('Loading in sentences...')
    # load in sentences, we need review id
    sentences = SentenceSplitter().split_reviews(reviews['text'], read_cache=use_splitted_cache, save_in_cache=False)

    logging.info('Selecting top N topics for each sentence...')
    # only keep the top n topics with the highest probability
    user_profiles = topic_distributions.apply(select_top_n, n=top_n_topics, axis=1)

    logging.info('Aggregating sentences by review...')
    # add the review id to the data, so we can concatenate the sentences and aggregate (sum) them per review
    user_profiles = pd.concat([sentences['review_id'], user_profiles], axis=1)
    user_profiles = user_profiles.groupby('review_id').aggregate('sum')

    logging.info('Aggregating reviews by user...')
    # add the user id to the data, so we can concatenate the reviews and aggregate (sum) them per user
    user_profiles = pd.concat([reviews['user_id'], user_profiles], axis=1)
    user_profiles = user_profiles.groupby('user_id').aggregate('sum')

    logging.info('Normalizing user profiles...')
    # normalize the user profiles -> [0,1]
    user_profiles = user_profiles.apply(normalize_user_profile, axis=1)

    # if no topic is relevant for any review from a user, the userprofile will be [0,0,...,0]
    # should basically never happen
    user_profiles = user_profiles.fillna(0)

    # save the user_profile for later use
    logging.info('Saving user profiles...')
    nlp_cache.save_user_profiles(user_profiles, name=profile_name)
    logging.info(f'Saved user profiles with name: {profile_name}')


def main_user_profile_topic(reviews: pd.DataFrame, amount_of_batches: int = 10,
                            profile_name: str = "BASIC_USER_PROFILES.parquet", use_cache: bool = True,
                            scores_save_dir: str = "base", model_name: str = None):
    logging.info('Finished reading in data, starting NLP...')
    nlp_cache = NLPCache()

    if not use_cache or not nlp_cache.is_available_scores(scores_save_dir):
        logging.warning(
            f'Cache is not being used: allowed: {use_cache} - available: {nlp_cache.is_available_scores(scores_save_dir)}')
        logging.info('Calculating bert_scores...')
        for index, batch in enumerate(tqdm(np.array_split(reviews, amount_of_batches), desc="Score Batches")):
            print()
            scores = create_scores_from_online_model_by_topic(batch['text'], use_cache=False, save_in_cache=False,
                                                              early_return=True, model_name=model_name)
            scores.columns = [str(x) for x in scores.columns]
            scores.to_parquet(nlp_cache.scores_path.joinpath(Path(scores_save_dir, f"score_part_{index}.parquet"),
                                                             engine='fastparquet'))

    logging.info('Loading in all scores...')
    scores = nlp_cache.load_scores(scores_save_dir)

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
