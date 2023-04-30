import pandas as pd
import numpy as np
import logging
from pathlib import Path
from bertopic import BERTopic
from tqdm import tqdm
from NLP.ModelsImplementations.CustomBERTopic import CustomBERTTopic
from NLP.df_NLP_manipulation.df_sentiment_analysis import sentiment_analysis_sentences
from NLP.managers.nlp_cache_manager import NLPCache
from NLP.managers.nlp_model_manager import NLPModels
from NLP.utils.scoring_functions import online_bertopic_scoring_func
from NLP.utils.sentence_splitter import SentenceSplitter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer


def _load_model(model_name: str, verbose: bool):
    logging.info("Loading in model...")
    model_manager = NLPModels()

    model_online_BERTopic: BERTopic = model_manager.load_model(model_name)
    model_online_BERTopic.verbose = verbose
    model_online_BERTopic.calculate_probabilities = False

    return model_online_BERTopic


def create_scores_by_approximate_distribution(reviews: pd.Series, model_name: str = None, use_cache: bool = True,
                                              save_in_cache: bool = False, verbose: bool = True):
    # load in model
    model_online_BERTopic: BERTopic = _load_model(model_name, verbose)

    # split reviews into sentences
    logging.info('Splitting Sentences...')
    sentence_splitter = SentenceSplitter(verbose=verbose)
    reviews = sentence_splitter.split_reviews(reviews, read_cache=use_cache, save_in_cache=save_in_cache)

    logging.info('Approximating topic distributions...')
    topic_distributions, _ = model_online_BERTopic.approximate_distribution(reviews['text'])
    topic_distributions = pd.DataFrame(topic_distributions)

    return topic_distributions


def create_scores_from_online_model_by_topic(reviews: pd.Series, model_name: str = None, use_cache: bool = True,
                                             save_in_cache: bool = False, verbose: bool = True,
                                             early_return: bool = False, calculate_sentiment: bool = False):
    # load in model
    model_online_BERTopic: BERTopic = _load_model(model_name, verbose)

    # split reviews into sentences
    logging.info('Splitting Sentences...')
    sentence_splitter = SentenceSplitter(verbose=verbose)
    reviews = sentence_splitter.split_reviews(reviews, read_cache=use_cache, save_in_cache=save_in_cache)

    nlp_cache = NLPCache()

    logging.info('Calculating Topics...')
    topics, _ = model_online_BERTopic.transform(reviews['text'])

    logging.info('Saving Topics...')
    topics = pd.DataFrame(topics)
    topics.columns = [str(x) for x in topics.columns]
    topics.to_parquet(nlp_cache.scores_path.joinpath(Path(f"topics_tmp.parquet")), engine='fastparquet')

    if calculate_sentiment:
        logging.info('Calculating sentiment...')
        # sentiment label+score for each sentence
        reviews = sentiment_analysis_sentences(reviews, verbose=verbose)

        logging.info('Saving Sentiment...')
        # no need to save the text
        reviews = reviews.drop('text', axis=1)
        reviews.columns = [str(x) for x in reviews.columns]
        reviews.to_parquet(nlp_cache.scores_path.joinpath(Path(f"sentiment_tmp.parquet"), engine='fastparquet'))

    logging.info('Merging Dataframe with topics...')
    # add topics them to the dataframe
    col_names = list(reviews.columns) + ['topic_id']
    reviews = pd.concat([reviews, topics], axis=1)
    reviews.columns = col_names

    # return before creating the actual bert_scores: Useful for saving data early to be reused
    if early_return:
        if calculate_sentiment:
            return reviews[['review_id', 'topic_id', 'label_sentiment', 'score_sentiment']]
        return reviews[['review_id', 'topic_id']]

    assert calculate_sentiment, """sentiment must be calculated when directly calculating user_profile scores.
                                Recommended is using an early return and using a precalculated sentiment in base"""

    # merge sentences back to one review
    reviews = reviews.groupby('review_id').aggregate(lambda item: item.tolist())
    # convert elements to numpy array
    reviews[['topic_id', 'label_sentiment', 'score_sentiment']] = reviews[
        ['topic_id', 'label_sentiment', 'score_sentiment']].applymap(
        np.array)

    logging.info('calculating final scores...')

    # aggregate scores using a custom formula
    bert_scores = reviews[
        ['topic_id', 'label_sentiment', 'score_sentiment']].apply(
        online_bertopic_scoring_func, total_amount_topics=len(model_online_BERTopic.get_topic_info()['Topic']), axis=1)

    # return each feature in a separate
    return pd.DataFrame(bert_scores.to_list())


def create_model_online_BERTopic(reviews: pd.Series, sentence_batch_size: int = 500_000, model_name: str = None, dim_red_components: int = 15,
                                 max_topics: int = 200, guided_topics: list[list[str]] = None):
    # split reviews into sentences
    logging.info("Splitting Sentences...")
    sentence_splitter = SentenceSplitter()
    reviews = sentence_splitter.split_reviews(reviews)
    input_data = reviews['text']

    logging.info("Doing Online BERTopic...")
    # Prepare sub-models that support online learning
    # keep X features (for every 786 vector)
    online_dim_reduction = IncrementalPCA(n_components=dim_red_components)
    online_clustering = MiniBatchKMeans(n_clusters=max_topics, random_state=0, batch_size=2048)
    # low decay because we want to keep as much data
    online_vectorizer = OnlineCountVectorizer(stop_words="english", decay=.01)

    BERTopic_online = CustomBERTTopic(max_topics=max_topics, dim_reduction_model=online_dim_reduction,
                                      cluster_model=online_clustering, vectorizer_model=online_vectorizer,
                                      guided_topics=guided_topics)
    BERTopic_online_model = BERTopic_online.model

    model_manager = NLPModels()

    # total amount of sentences // amount of sentences per batch
    amount_of_batches = 1 + input_data.size // sentence_batch_size
    for batch in tqdm(np.array_split(input_data, amount_of_batches), desc="BERT Batches"):
        print()
        batch = batch.reset_index()['text']
        try:
            BERTopic_online_model.partial_fit(batch)
        except ValueError as ex:
            logging.info(f"SKIPPED BATCH: {ex}")
            continue
        model_manager.save_model(BERTopic_online_model, "online_model_tmp.bert")

    logging.info("Saving model...")
    model_manager.save_model(BERTopic_online_model, model_name)
