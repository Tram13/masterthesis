import pathlib

import numpy as np
import pandas as pd
import torch

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from NLP.ModelsImplementations.SBERT_feature_extraction import SentenceBERT


class CustomBERTTopic:

    def __init__(self, embedding_model=None, dim_reduction_model=None, cluster_model=None, vectorizer_model=None,
                 ctfidf_model=None, fine_tuning_representation_model=KeyBERTInspired(), max_topics="auto", batch_size: int = 32,
                 guided_topics=None):
        self.batch_size = batch_size
        # Extract embeddings
        # see docs for other models
        self.embedding_model = SentenceBERT().model if embedding_model is None else embedding_model

        # Reduce dimensionality
        # MUST SUPPORT: `.fit` and `.transform` functions.
        self.dim_reduction_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0,
                                        metric='cosine') if dim_reduction_model is None else dim_reduction_model

        # Cluster reduced embeddings
        # MUST SUPPORT: .fit` and `.predict` functions along with the `.labels_` variable.
        self.cluster_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom',
                                     prediction_data=False) if cluster_model is None else cluster_model

        # Tokenize clusters => no assumptions of cluster structure
        self.vectorizer_model = CountVectorizer(stop_words="english", min_df=3) if vectorizer_model is None else vectorizer_model

        # Cluster - TF-IDF
        self.ctfidf_model = ClassTfidfTransformer() if ctfidf_model is None else ctfidf_model

        # [OPTIONAL] Fine-tune topic representations
        # any `bertopic.representation` model
        self.fine_tuning_representation_model = fine_tuning_representation_model

        self.guided_topics = guided_topics

        # All steps together to create the model
        self.model = BERTopic(
            verbose=True,
            nr_topics=max_topics,
            embedding_model=embedding_model,
            umap_model=dim_reduction_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=fine_tuning_representation_model,
            seed_topic_list=guided_topics
        )

        self.model.embedding_model = self.embedding_model
        self.model.umap_model = self.dim_reduction_model
        self.model.hdbscan_model = self.cluster_model
        self.model.vectorizer_model = self.vectorizer_model
        self.model.ctfidf_model = self.ctfidf_model
        self.model.representation_model = self.fine_tuning_representation_model

        self.device = self.enable_gpu()

        print(f'current device: {self.model.embedding_model.device}')

    def enable_gpu(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.embedding_model.to(device)
        return device

    def precompute_and_save_embeddings(self, data: pd.Series, save_path: pathlib.Path = None) -> np.ndarray:
        if isinstance(self.model.embedding_model, SentenceTransformer):
            embeddings = self.model.embedding_model.encode(sentences=data, show_progress_bar=True, device='cuda', batch_size=self.batch_size)
        else:
            raise NotImplementedError()

        if save_path is not None:
            np.save(file=save_path.with_suffix('.npy'), arr=embeddings)

        return embeddings
