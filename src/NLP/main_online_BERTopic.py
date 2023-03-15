import pandas as pd
import numpy as np
import logging
from pathlib import Path
from bertopic import BERTopic
from tqdm import tqdm
from src.NLP.ModelsImplementations.CustomBERTopic import CustomBERTTopic
from src.NLP.df_NLP_manipulation.df_sentiment_analysis import sentiment_analysis_sentences
from src.NLP.scoring_functions import online_bertopic_scoring_func
from src.NLP.sentence_splitter import SentenceSplitter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer
from src.tools.config_parser import ConfigParser


def create_scores_from_online_model(reviews: pd.Series, current_model_save_path: str = None, use_cache: bool = True,
                                    save_in_cache: bool = False, verbose: bool = True, early_return: bool = False):
    logging.info("Loading in model...")
    if current_model_save_path is None:
        current_save_dir = Path(ConfigParser().get_value('data', 'online_bert_model_path'))
        if not current_save_dir.is_dir():
            current_save_dir.mkdir()
        current_model_save_path = current_save_dir.joinpath(
            Path(ConfigParser().get_value('data', 'use_bert_model_fname')))

    if not current_model_save_path.is_file():
        raise FileNotFoundError("No model found")

    model_online_BERTopic: BERTopic = BERTopic.load(current_model_save_path)
    model_online_BERTopic.verbose = verbose
    model_online_BERTopic.calculate_probabilities = False

    # split reviews into sentences
    logging.info('Splitting Sentences...')
    sentence_splitter = SentenceSplitter(verbose=verbose)
    reviews = sentence_splitter.split_reviews(reviews, read_cache=use_cache, save_in_cache=save_in_cache)

    logging.info('Calculating Topics')
    topics, _ = model_online_BERTopic.transform(reviews['text'])

    logging.info('Calculating sentiment...')
    # sentiment label+score for each sentence
    reviews = sentiment_analysis_sentences(reviews, verbose=verbose)

    logging.info('Merging Dataframe...')
    # add them to the dataframe
    col_names = list(reviews.columns) + ['topic_id']
    reviews = pd.concat([reviews, pd.Series(topics)], axis=1)
    reviews.columns = col_names

    if early_return:
        return reviews[['review_id', 'topic_id', 'label_sentiment', 'score_sentiment']]

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


def create_model_online_BERTopic(reviews: pd.Series, sentence_batch_size: int = 500_000, model_name: str = None,
                                 max_topics: int = 100):
    # split reviews into sentences
    print('Splitting Sentences...')
    sentence_splitter = SentenceSplitter()
    reviews = sentence_splitter.split_reviews(reviews)
    input_data = reviews['text']

    print('Doing Online BERTopic...')
    # Prepare sub-models that support online learning
    # keep 15 features (for every 786 vector)
    online_dim_reduction = IncrementalPCA(n_components=15)
    online_clustering = MiniBatchKMeans(n_clusters=50, random_state=0, batch_size=2048)
    # low decay because we want to keep as much data
    online_vectorizer = OnlineCountVectorizer(stop_words="english", decay=.01)

    BERTopic_online = CustomBERTTopic(max_topics=max_topics, dim_reduction_model=online_dim_reduction,
                                      cluster_model=online_clustering, vectorizer_model=online_vectorizer)
    BERTopic_online_model = BERTopic_online.model

    current_save_dir = Path(ConfigParser().get_value('data', 'online_bert_model_path'))
    if not current_save_dir.is_dir():
        current_save_dir.mkdir()

    tmp_save_path = current_save_dir.joinpath("online_model_tmp.bert")

    # total amount of sentences // amount of sentences per batch
    amount_of_batches = 1 + input_data.size // sentence_batch_size
    for batch in tqdm(np.array_split(input_data, amount_of_batches), desc="BERT Batches"):
        print()
        batch = batch.reset_index()['text']
        BERTopic_online_model.partial_fit(batch)
        BERTopic_online_model.save(tmp_save_path)

    if model_name is None:
        model_name = Path(ConfigParser().get_value('data', 'online_bert_model_fname'))
    current_save_path = current_save_dir.joinpath(model_name)

    BERTopic_online_model.save(current_save_path)
    print(f'Saved final model in: {current_save_path}')
