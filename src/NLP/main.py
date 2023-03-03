import pandas as pd
import numpy as np

from pathlib import Path

from bertopic import BERTopic
from tqdm import tqdm
from umap import UMAP

from src.NLP.ClusteringMetrics import ClusteringMetrics
from src.NLP.ModelsImplementations.CustomBERTopic import CustomBERTTopic
from src.NLP.df_NLP_manipulation.df_clustering import cluster_sentences
from src.NLP.df_NLP_manipulation.df_sentiment_analysis import sentiment_analysis_sentences
from src.NLP.scoring_functions import bertopic_scoring_func, basic_clustering_scoring_func, online_bertopic_scoring_func
from src.NLP.sentence_splitter import SentenceSplitter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer

from src.tools.config_parser import ConfigParser


def create_scores_from_online_model(reviews: pd.Series, current_model_save_path: str = None):
    print("Loading in model...")
    if current_model_save_path is None:
        current_save_dir = Path(ConfigParser().get_value('data', 'online_bert_model_path'))
        current_model_save_path = current_save_dir.joinpath(Path(ConfigParser().get_value('data', 'online_bert_model_fname')))

    if not current_model_save_path.is_file():
        raise FileNotFoundError("No model found")

    model_online_BERTopic: BERTopic = BERTopic.load(current_model_save_path)
    model_online_BERTopic.verbose = True

    # split reviews into sentences
    print('Splitting Sentences...')
    sentence_splitter = SentenceSplitter()
    reviews = sentence_splitter.split_reviews(reviews)

    print('Calculating Topics')
    topics, _ = model_online_BERTopic.transform(reviews['text'])

    print('Calculating sentiment...')
    # sentiment label+score for each sentence
    reviews = sentiment_analysis_sentences(reviews)

    print('Merging Dataframe...')
    # add them to the dataframe
    col_names = list(reviews.columns) + ['topic_id']
    reviews = pd.concat([reviews, pd.Series(topics)], axis=1)
    reviews.columns = col_names
    # merge sentences back to one review
    reviews = reviews.groupby('review_id').aggregate(lambda item: item.tolist())
    # convert elements to numpy array
    reviews[['topic_id', 'label_sentiment', 'score_sentiment']] = reviews[
        ['topic_id', 'label_sentiment', 'score_sentiment']].applymap(
        np.array)

    print('calculating final scores...')
    # aggregate scores using a custom formula
    bert_scores = reviews[
        ['topic_id', 'label_sentiment', 'score_sentiment']].apply(
        online_bertopic_scoring_func, total_amount_topics=len(model_online_BERTopic.get_topic_info()['Topic']), axis=1)

    return bert_scores


def create_model_online_BERTopic(reviews: pd.Series, sentence_batch_size: int = 500_000):
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

    BERTopic_online = CustomBERTTopic(max_topics=100, dim_reduction_model=online_dim_reduction, cluster_model=online_clustering, vectorizer_model=online_vectorizer)
    BERTopic_online_model = BERTopic_online.model

    # total amount of sentences // amount of sentences per batch
    amount_of_batches = 1 + input_data.size // sentence_batch_size
    for batch in tqdm(np.array_split(input_data, amount_of_batches), desc="BERT Batches"):
        print()
        batch = batch.reset_index()['text']
        BERTopic_online_model.partial_fit(batch)

    current_save_dir = Path(ConfigParser().get_value('data', 'online_bert_model_path'))
    if not current_save_dir.is_dir():
        current_save_dir.mkdir()

    current_save_path = current_save_dir.joinpath(Path(ConfigParser().get_value('data', 'online_bert_model_fname')))

    BERTopic_online_model.save(current_save_path)
    print(f'Saved final model in: {current_save_path}')


def main_BERTopic(reviews: pd.Series, embeddings: np.ndarray = None, do_precompute_and_save_embeddings: bool = False,
                  save_path: Path = None) -> tuple[pd.Series, pd.DataFrame]:
    # split reviews into sentences
    print('Splitting Sentences...')
    sentence_splitter = SentenceSplitter()
    reviews = sentence_splitter.split_reviews(reviews)
    input_data = reviews['text']

    # create the model
    BERTopic = CustomBERTTopic(max_topics=50, batch_size=32)
    BERTopic_model = BERTopic.model

    print('Creating Embeddings...')
    # if there are no given embeddings, it is possible to precompute them and save them
    if embeddings is None and do_precompute_and_save_embeddings:
        embeddings = BERTopic.precompute_and_save_embeddings(input_data, save_path=save_path)

    # generate topics
    topics, probabilities = BERTopic_model.fit_transform(input_data, embeddings=embeddings)
    force_topics = BERTopic_model.reduce_outliers(documents=input_data, topics=topics, threshold=0.1)

    print("finished BERTopic, calculating metrics")

    # clustering Metrics
    clustering_metric_default = ClusteringMetrics(embeddings, topics)
    clustering_metric_force_topic = ClusteringMetrics(embeddings, force_topics)

    clustering_metric_default.calculate_all_indices()
    clustering_metric_force_topic.calculate_all_indices()

    print(f'Metrics for default: {str(clustering_metric_default)}\n')
    print(f'Metrics for forced: {str(clustering_metric_force_topic)}\n')

    # add them to the dataframe
    col_names = list(reviews.columns) + ['topic_id', 'force_topic_id', 'topic_probability']
    reviews = pd.concat([reviews, pd.Series(topics), pd.Series(force_topics), pd.Series(probabilities)], axis=1)
    reviews.columns = col_names

    print('calculating sentiment...')
    # sentiment label+score for each sentence
    reviews = sentiment_analysis_sentences(reviews)

    # merge sentences back to one review
    reviews = reviews.groupby('review_id').aggregate(lambda item: item.tolist())

    # convert elements to numpy array
    reviews[['topic_id', 'force_topic_id', 'topic_probability', 'label_sentiment', 'score_sentiment']] = reviews[
        ['topic_id', 'force_topic_id', 'topic_probability', 'label_sentiment', 'score_sentiment']].applymap(
        np.array)

    print('calculating final scores...')

    # aggregate scores using a custom formula
    bert_scores = reviews[
        ['topic_id', 'force_topic_id', 'topic_probability', 'label_sentiment', 'score_sentiment']].apply(
        bertopic_scoring_func, total_amount_topics=len(BERTopic_model.get_topic_info()['Topic']),
        weight_main_topics=0.75, axis=1)

    return bert_scores, BERTopic_model.get_topic_info()


def main_basic_clustering(reviews: pd.Series):
    # split reviews into sentences
    sentence_splitter = SentenceSplitter()
    splitted_reviews = sentence_splitter.split_reviews(reviews)

    # clustering label
    reviews, amount_clusters = cluster_sentences(splitted_reviews)

    # sentiment label+score for each sentence
    reviews = sentiment_analysis_sentences(reviews)

    # merge sentences back to one review
    reviews = reviews.groupby('review_id').aggregate(lambda item: item.tolist())

    # convert elements to numpy array
    reviews[['label_sentiment', 'score_sentiment']] = reviews[['label_sentiment', 'score_sentiment']].applymap(np.array)

    # aggregate scores using a custom formula based on cluster and sentiment
    reviews['cluster_scores'] = reviews[['cluster_labels', 'label_sentiment', 'score_sentiment']].apply(
        basic_clustering_scoring_func, total_amount_topics=amount_clusters, axis=1
    )

    return reviews  # todo niet alles moet gereturned worden


def test_manual_bert(reviews: pd.Series):
    # split reviews into sentences
    print('Splitting Sentences...')
    sentence_splitter = SentenceSplitter()
    reviews = sentence_splitter.split_reviews(reviews)
    input_data = reviews['text']

    # create the model
    BERTopic = CustomBERTTopic(max_topics=50, batch_size=32)

    print('Creating Embeddings...')
    # if there are no given embeddings, it is possible to precompute them and save them
    embeddings = BERTopic.precompute_and_save_embeddings(input_data, save_path=None)

    # generate topics
    dim_reduction_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                               metric='cosine')
    embeddings = dim_reduction_model.transform(embeddings)


if __name__ == '__main__':
    create_scores_from_online_model()


    # bert_scores, bert_topics = main_BERTopic(reviews_input_big, do_precompute_and_save_embeddings=True)
    # print(bert_scores)
