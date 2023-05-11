import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from NLP.managers.nlp_cache_manager import NLPCache
from NLP.managers.nlp_model_manager import NLPModels
from NLP.utils.ClusteringMetrics import ClusteringMetrics


def evaluate_model(sentences, model_name, percentage, divide_10=False, dim_reduction=True):

    nlp_cache = NLPCache(amount_of_embeddings_batches=percentage)
    nlp_models = NLPModels()
    bertopic_model = nlp_models.load_model(model_name)

    if not nlp_cache.is_available_embeddings():
        logging.info('Creating embeddings...')

        for index, batch in enumerate(tqdm(np.array_split(sentences, 100), desc="Embedding batches")):
            print()
            batch = batch.reset_index()['text']
            features = bertopic_model.embedding_model.embed_documents(batch, verbose=True)
            features = pd.DataFrame(features)
            features.columns = [str(x) for x in features.columns]
            nlp_cache.save_embeddings(features, index)

    logging.info('Loading in embeddings...')
    features = nlp_cache.load_embeddings(percentage)

    if divide_10:
        features = features.head(len(features.index)//10)

    if dim_reduction:
        # also apply dimensionality reduction to our embedding
        features = bertopic_model.umap_model.transform(features)

    logging.info('Loading in topics...')
    topics = np.array(nlp_cache.load_scores(nlp_models.get_dir_for_model(model_name))['topic_id'])[:features.shape[0]]

    logging.info('Ready to calculate clustering metrics')
    metrics = ClusteringMetrics(features=np.array(features), labels=topics)

    with open('metrics_TMP3.csv', 'a') as f:
        logging.info("calculate_calinski_harabasz_score")
        metrics.calculate_calinski_harabasz_score()
        f.write(f"{metrics.calinski_harabasz_score},")

        logging.info("calculate_davies_bouldin_index")
        metrics.calculate_davies_bouldin_index()
        f.write(f"{metrics.davies_bouldin_index},")

        # logging.info("calculate_dunn_index")
        # metrics.calculate_dunn_index()
        # f.write(f"{metrics.dunn_index},")

        logging.info("calculate_silhouet_index")
        metrics.calculate_silhouet_index()
        f.write(f"{metrics.silhouet_index}\n")

