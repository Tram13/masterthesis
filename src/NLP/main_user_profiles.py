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

tqdm.pandas()


def main_user_profile_approximation(reviews: pd.DataFrame, amount_of_batches_for_approximations: int = 1,
                                    model_name: str = None, amount_of_batches_top_n: int = 10,
                                    profile_name: str = None, use_cache: bool = True,
                                    use_splitted_cache: bool = True, top_n_topics: int = 5,
                                    approx_save_dir: str = "base", filter_select: list[int] = None,
                                    normalize_after_selection: bool = False):
    if profile_name is None:
        profile_name = f"APPROX_USER_PROFILES_top_{top_n_topics}_normalize_{normalize_after_selection}.parquet"

    logging.info('Finished reading in data, starting NLP...')
    nlp_cache = NLPCache(amount_of_approximation_batches=amount_of_batches_for_approximations,
                         amount_of_top_n_batches=amount_of_batches_top_n)

    if not use_cache or not nlp_cache.is_available_approximation(approx_save_dir):
        logging.warning(
            f'Cache is not being used for approximations: allowed: {use_cache} - available: {nlp_cache.is_available_approximation(approx_save_dir)}')
        logging.info('Calculating approximations for current model...')
        for index, batch in enumerate(
                tqdm(np.array_split(reviews, amount_of_batches_for_approximations), desc="Approximation batches")):
            print()
            approximation = create_scores_by_approximate_distribution(batch['text'], model_name=model_name,
                                                                      use_cache=amount_of_batches_for_approximations == 1)
            approximation.columns = [str(x) for x in approximation.columns]
            approximation.to_parquet(
                nlp_cache.approximation_path.joinpath(Path(approx_save_dir, f"approximation_part_{index}.parquet")),
                engine='fastparquet')

    logging.info('Loading in approximations...')
    user_profiles = nlp_cache.load_approximation(approx_save_dir)

    # select only topics that are relevant and not too general
    if filter_select:
        user_profiles = user_profiles[filter_select]

    logging.info('Loading in sentences...')
    # load in sentences, we need review id
    sentences = SentenceSplitter().split_reviews(reviews['text'], read_cache=use_splitted_cache, save_in_cache=False)

    # only keep the top n topics with the highest probability
    if not use_cache or not nlp_cache.is_available_top_n(top_n_topics, approx_save_dir,
                                                         normalized=normalize_after_selection):
        logging.warning(
            f'Cache is not being used for selecting top n with n={top_n_topics}: allowed: {use_cache} - available: {nlp_cache.is_available_top_n(top_n_topics, approx_save_dir, normalized=normalize_after_selection)}')
        logging.info('Selecting top N topics for each sentence...')
        if normalize_after_selection:
            logging.info('+ Normalizing top_n_topics...')
        for index, batch in enumerate(
                tqdm(np.array_split(user_profiles, amount_of_batches_top_n), desc="Top N batches")):
            top_n_selected = batch.progress_apply(select_top_n, n=top_n_topics, axis=1)
            if normalize_after_selection:
                top_n_selected = top_n_selected.progress_apply(normalize_user_profile, axis=1)
            top_n_selected.columns = [str(x) for x in top_n_selected.columns]
            nlp_cache.save_top_n_filter(top_n_selected, n=top_n_topics, index=index, save_dir=approx_save_dir,
                                        normalized=normalize_after_selection)

    logging.info('Collecting top_n_topics...')
    user_profiles = nlp_cache.load_top_n_filter(n=top_n_topics, save_dir=approx_save_dir,
                                                normalized=normalize_after_selection)

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
    user_profiles = user_profiles.progress_apply(normalize_user_profile, axis=1)

    # if no topic is relevant for any review from a user, the userprofile will be [0,0,...,0]
    # should basically never happen
    user_profiles = user_profiles.fillna(0)

    # save the user_profile for later use
    logging.info('Saving user profiles...')
    nlp_cache.save_user_profiles(user_profiles, name=profile_name)
    logging.info(f'Saved user profiles with name: {profile_name}')


def main_user_profile_topic(reviews: pd.DataFrame, amount_of_batches: int = 10,
                            profile_name: str = "BASIC_USER_PROFILES.parquet", use_cache: bool = True,
                            scores_save_dir: str = "base", model_name: str = None,
                            use_sentiment_in_scores: bool = False):
    logging.info('Finished reading in data, starting NLP...')
    nlp_cache = NLPCache(amount_of_scores_batches=amount_of_batches)
    nlp_models = NLPModels()

    if model_name:
        scores_save_dir = nlp_models.get_dir_for_model(model_name)

    if not use_cache or not nlp_cache.is_available_scores(scores_save_dir):
        logging.warning(
            f'Cache is not being used: allowed: {use_cache} - available: {nlp_cache.is_available_scores(scores_save_dir)}')
        logging.info('Calculating bert_scores...')
        for index, batch in enumerate(tqdm(np.array_split(reviews, amount_of_batches), desc="Score Batches")):
            print()
            scores = create_scores_from_online_model_by_topic(batch['text'], use_cache=False, model_name=model_name,
                                                              save_in_cache=False, early_return=True)
            scores.columns = [str(x) for x in scores.columns]
            nlp_cache.save_scores(scores, index, scores_save_dir)

    logging.info('Loading in all scores...')
    scores = nlp_cache.load_scores(scores_save_dir)

    # sentiment is only included in the base scores
    if scores_save_dir != "base" and use_sentiment_in_scores:
        sentiment = nlp_cache.load_sentiment()
        scores = pd.concat([sentiment, scores["topic_id"]], axis=1)

    columns_to_use = ['topic_id', 'label_sentiment', 'score_sentiment'] if use_sentiment_in_scores else ['topic_id']

    # merge sentences back to one review
    logging.info('Merging Reviews...')

    scores = scores.groupby('review_id').aggregate(lambda item: item.tolist())
    # convert elements to numpy array
    scores[columns_to_use] = scores[columns_to_use].applymap(np.array)

    logging.info("Loading in model...")
    model_manager = NLPModels()
    model_online_BERTopic: BERTopic = model_manager.load_model(model_name)

    logging.info("Calculating bert_scores...")
    bert_scores = scores[columns_to_use].apply(
        online_bertopic_scoring_func, total_amount_topics=len(model_online_BERTopic.get_topic_info()['Topic']),
        use_sentiment=use_sentiment_in_scores, axis=1)

    logging.info('creating user profiles from bert_scores...')
    if use_sentiment_in_scores:
        logging.info("Exploding bert_scores...")
        bert_scores = pd.DataFrame(bert_scores.to_list())
        user_profiles = calculate_basic_user_profiles(reviews, bert_scores)
    else:
        user_profiles = calculate_basic_user_profiles(reviews, bert_scores, 'sum')

        logging.info("Exploding bert_scores (late) & normalizing user profiles...")
        user_profiles = pd.concat([user_profiles['user_id'],
                                   pd.DataFrame(user_profiles[0].to_list()).progress_apply(normalize_user_profile,
                                                                                           axis=1)], axis=1)

    user_profiles.columns = [str(x) for x in user_profiles.columns]

    logging.info('Saving user profiles...')
    user_profiles.to_parquet(nlp_cache.user_profiles_path.joinpath(Path(profile_name)), engine='fastparquet')
