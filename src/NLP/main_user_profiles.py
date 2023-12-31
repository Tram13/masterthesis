import contextlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import swifter
from bertopic import BERTopic
from tqdm import tqdm

from NLP.main_online_BERTopic import create_scores_from_online_model_by_topic, create_scores_by_approximate_distribution
from NLP.managers.nlp_cache_manager import NLPCache
from NLP.managers.nlp_model_manager import NLPModels
from NLP.utils.scoring_functions import online_bertopic_scoring_func
from NLP.utils.sentence_splitter import SentenceSplitter
from NLP.utils.user_profile_creation import select_top_n, normalize_user_profile, calculate_basic_user_profiles_from_dataframe, \
    calculate_basic_user_profiles_from_series

# Trust me bro
with contextlib.redirect_stdout(None):
    _ = swifter.config


def main_user_profile_approximation(reviews: pd.DataFrame, amount_of_batches_for_approximations: int = 1,
                                    model_name: str = None, amount_of_batches_top_n: int = 10,
                                    profile_name: str = None, use_cache: bool = True,
                                    use_splitted_cache: bool = True, top_n_topics: int = 5,
                                    use_sentiment_in_scores: bool = False,
                                    approx_save_dir: str = "base", filter_select: list[str] = None,
                                    normalize_after_selection: bool = False, profile_mode: str = "user_id",
                                    part_of_dataset: bool = False):
    if profile_name is None:
        profile_name = f"APPROX_USER_PROFILES_top_{top_n_topics}_normalize_{normalize_after_selection}.parquet"

    nlp_models = NLPModels()
    if model_name:
        approx_save_dir = nlp_models.get_dir_for_model(model_name)

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
            nlp_cache.save_approximation(approximation, index=index, model_dir=approx_save_dir)


    logging.info('Loading in approximations...')
    user_profiles = nlp_cache.load_approximation(approx_save_dir)

    # select only topics that are relevant and not too general
    filter_string = ""
    if filter_select:
        filter_string = "_USER" if profile_mode == "user_id" else "_BUSINESS"
        user_profiles = user_profiles[filter_select]

    logging.info('Loading in sentences...')
    # load in sentences, we need review id
    sentences = SentenceSplitter().split_reviews(reviews['text'], read_cache=use_splitted_cache, save_in_cache=False)

    # only keep the top n topics with the highest probability
    if not use_cache or not nlp_cache.is_available_top_n(top_n_topics, approx_save_dir,
                                                         normalized=normalize_after_selection,
                                                         filter_string=filter_string,
                                                         sentiment=use_sentiment_in_scores):
        logging.warning(
            f'Cache is not being used for selecting top n with n={top_n_topics},{normalize_after_selection=},{filter_string},{use_sentiment_in_scores}: allowed: {use_cache} - available: {nlp_cache.is_available_top_n(top_n_topics, approx_save_dir, normalized=normalize_after_selection, filter_string=filter_string, sentiment=use_sentiment_in_scores)}')
        logging.info('Selecting top N topics for each sentence...')
        if normalize_after_selection:
            logging.info('+ Normalizing top_n_topics...')
        for index, batch in enumerate(
                tqdm(np.array_split(user_profiles, amount_of_batches_top_n), desc="Top N batches")):
            top_n_selected = batch.swifter.apply(select_top_n, n=top_n_topics, axis=1)
            if normalize_after_selection:
                top_n_selected = top_n_selected.swifter.apply(normalize_user_profile, axis=1)
            if use_sentiment_in_scores:
                sentiment = nlp_cache.load_sentiment()["label_sentiment"]
                top_n_selected = top_n_selected.multiply(sentiment, axis=0)
            top_n_selected.columns = [str(x) for x in top_n_selected.columns]
            nlp_cache.save_top_n_filter(top_n_selected, n=top_n_topics, index=index, save_dir=approx_save_dir,
                                        normalized=normalize_after_selection, filter_string=filter_string,
                                        sentiment=use_sentiment_in_scores)

    logging.info('Collecting top_n_topics...')
    user_profiles = nlp_cache.load_top_n_filter(n=top_n_topics, save_dir=approx_save_dir,
                                                normalized=normalize_after_selection, filter_string=filter_string,
                                                sentiment=use_sentiment_in_scores)

    logging.info('Aggregating sentences by review...')
    # add the review id to the data, so we can concatenate the sentences and aggregate (sum) them per review
    user_profiles = pd.concat([sentences['review_id'], user_profiles], axis=1)
    user_profiles = user_profiles.groupby('review_id').aggregate('sum')

    # only select reviews we can use
    if part_of_dataset:
        user_profiles = user_profiles.loc[reviews.index]

    # normalize the review profiles -> [0,1]
    user_profiles = user_profiles.progress_apply(normalize_user_profile, axis=1)

    logging.info('Aggregating reviews by user_id or business_id...')
    # add the user id to the data, so we can concatenate the reviews and aggregate (sum) them per user

    user_profiles = reviews[[profile_mode]].join(user_profiles, on='review_id', how='inner').groupby(profile_mode).agg('sum')

    logging.info(f'Normalizing {profile_mode[:-3]} profiles...')
    # normalize the user profiles -> [0,1]
    user_profiles = user_profiles.swifter.apply(normalize_user_profile, axis=1)

    # if no topic is relevant for any review from a user, the userprofile will be [0,0,...,0]
    # should basically never happen
    user_profiles = user_profiles.fillna(0)

    # save the user_profile for later use
    logging.info(f'Saving {profile_mode[:-3]} profiles...')
    if profile_mode == 'user_id':
        nlp_cache.save_user_profiles(user_profiles, profile_name)
    else:
        nlp_cache.save_business_profiles(user_profiles, profile_name)
    logging.info(f'Saved user profiles with name: {profile_name}')

    return user_profiles


def main_user_profile_topic(reviews: pd.DataFrame, amount_of_batches: int = 10,
                            profile_name: str = "BASIC_USER_PROFILES.parquet", use_cache: bool = True,
                            scores_save_dir: str = "base", model_name: str = None,
                            use_sentiment_in_scores: bool = False, profile_mode: str = 'user_id',
                            part_of_dataset: bool = False, only_create_scores: bool = False,
                            calculate_sentiment: bool = False):
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
                                                              save_in_cache=False, early_return=True,
                                                              calculate_sentiment=calculate_sentiment)
            scores.columns = [str(x) for x in scores.columns]
            nlp_cache.save_scores(scores, index, scores_save_dir)

    logging.info('Loading in all scores...')
    scores = nlp_cache.load_scores(scores_save_dir)

    if only_create_scores:
        return

    # add sentiment
    if use_sentiment_in_scores:
        sentiment = nlp_cache.load_sentiment()
        scores = pd.concat([sentiment, scores["topic_id"]], axis=1)

    columns_to_use = ['topic_id', 'label_sentiment', 'score_sentiment'] if use_sentiment_in_scores else ['topic_id']

    # merge sentences back to one review
    logging.info('Merging Reviews...')

    scores = scores.groupby('review_id').aggregate(lambda item: item.tolist())

    # only select reviews we can use
    if part_of_dataset:
        scores = scores.loc[reviews.index]

    # convert elements to numpy array
    scores[columns_to_use] = scores[columns_to_use].applymap(np.array)

    logging.info("Loading in NLP model...")
    model_manager = NLPModels()
    model_online_BERTopic: BERTopic = model_manager.load_model(model_name)

    logging.info("Calculating bert_scores...")
    bert_scores = scores[columns_to_use].swifter.apply(
        online_bertopic_scoring_func, total_amount_topics=len(model_online_BERTopic.get_topic_info()['Topic']),
        use_sentiment=use_sentiment_in_scores, axis=1)

    logging.info('creating user profiles from bert_scores...')
    if use_sentiment_in_scores:
        logging.info("Exploding bert_scores...")
        bert_scores = pd.DataFrame(bert_scores.to_list())
        user_profiles = calculate_basic_user_profiles_from_dataframe(reviews, bert_scores, mode=profile_mode)
    else:
        user_profiles = calculate_basic_user_profiles_from_series(reviews, bert_scores, 'sum', mode=profile_mode)

        logging.info(f"Exploding bert_scores (late) & normalizing {profile_mode[:-3]} profiles...")
        user_profiles = pd.DataFrame([*user_profiles['bert_scores']], index=user_profiles.index).swifter.apply(normalize_user_profile, axis=1)

    user_profiles.columns = [str(x) for x in user_profiles.columns]

    logging.info(f'Saving {profile_mode[:-3]} profiles...')
    if profile_mode == 'user_id':
        nlp_cache.save_user_profiles(user_profiles, profile_name)
    else:
        nlp_cache.save_business_profiles(user_profiles, profile_name)
    logging.info(f'Saved {profile_mode[:-3]} profiles with name: {profile_name}')

    return user_profiles
