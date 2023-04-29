import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from NLP.managers.nlp_cache_manager import NLPCache
from NLP.managers.nlp_model_manager import NLPModels
from NLP.utils.ClusteringMetrics import ClusteringMetrics


def evaluate_model(sentences, model_name):
    amount_of_embedding_batches = 100
    nlp_cache = NLPCache(amount_of_embedding_batches)
    nlp_models = NLPModels()

    if not nlp_cache.is_available_embeddings():
        logging.info('Creating embeddings...')

        bertopic_model = nlp_models.load_model(model_name)
        for index, batch in enumerate(tqdm(np.array_split(sentences, amount_of_embedding_batches), desc="Embedding batches")):
            print()
            batch = batch.reset_index()['text']
            features = bertopic_model.embedding_model.embed_documents(batch, verbose=True)
            features = pd.DataFrame(features)
            features.columns = [str(x) for x in features.columns]
            nlp_cache.save_embeddings(features, index)

    logging.info('Loading in embeddings...')
    # todo might be too memory heavy or too slow => batching
    # for now keep first x%
    features = nlp_cache.load_embeddings()

    logging.info('Loading in topics...')
    #  keep first x% TODO EDIT THIS
    topics = np.array(nlp_cache.load_scores(nlp_models.get_dir_for_model(model_name))['topic_id'])[:len(features.index)]

    logging.info('Ready to calculate clustering metrics')
    metrics = ClusteringMetrics(features=np.array(features), labels=topics)
    logging.info('Calculating Calinski Harabasz score...')
    score = metrics.calculate_calinski_harabasz_score()
    print(f'calinski_score = {score}')
