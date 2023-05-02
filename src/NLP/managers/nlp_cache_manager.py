from pathlib import Path

import numpy as np

from tools.config_parser import ConfigParser
import pandas as pd
import os


class NLPCache:

    def __init__(self, amount_of_scores_batches: int = 10, amount_of_zero_shot_batches: int = 30,
                 amount_of_approximation_batches: int = 1, amount_of_top_n_batches: int = 10,
                 amount_of_embeddings_batches: int = 100):
        self.cache_path = Path(ConfigParser().get_value('cache', 'nlp_cache_dir'))
        self.user_profiles_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'user_profiles_dir')))
        self.business_profile_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'business_profiles_dir')))
        self.scores_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'scores_dir')))
        self.sentiment_path = self.scores_path
        self.approximation_path = self.scores_path
        self.zero_shot_classes_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'zero_shot_dir')))
        self.guided_topics_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'guided_topics')))
        self.embeddings_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'embeddings')))

        self._make_dirs()

        # config amount of batches that was done
        self._amount_of_scores_batches = amount_of_scores_batches
        self._amount_of_zero_shot_batches = amount_of_zero_shot_batches
        self._amount_of_approximation_batches = amount_of_approximation_batches
        self._amount_of_top_n_batches = amount_of_top_n_batches
        self._amount_of_embeddings_batches = amount_of_embeddings_batches

    def _make_dirs(self):
        self._create_path_if_not_exists(self.cache_path)
        self._create_path_if_not_exists(self.user_profiles_path)
        self._create_path_if_not_exists(self.business_profile_path)
        self._create_path_if_not_exists(self.scores_path)
        self._create_path_if_not_exists(self.sentiment_path)
        self._create_path_if_not_exists(self.approximation_path)
        self._create_path_if_not_exists(self.zero_shot_classes_path)
        self._create_path_if_not_exists(self.guided_topics_path)
        self._create_path_if_not_exists(self.embeddings_path)

    @staticmethod
    def _create_path_if_not_exists(path: Path):
        if not path.is_dir():
            path.mkdir()

    def read_guided_topics(self, name: str = "NLP_categories.txt"):
        with open(self.guided_topics_path.joinpath(Path(name)), 'r') as f:
            return [line.strip().split(';') for line in f.readlines()]

    def save_embeddings(self, embeddings, index):
        embeddings.to_parquet(Path(self.embeddings_path, f"embedding_part_{index}.parquet"), engine='fastparquet')

    def save_business_profiles(self, business_profiles: pd.DataFrame, name: str = "BASIC_BUSINESS_PROFILES.parquet"):
        business_profiles.to_parquet(Path(self.business_profile_path, name), engine='fastparquet')

    def save_user_profiles(self, user_profiles: pd.DataFrame, name: str = "BASIC_USER_PROFILES.parquet"):
        user_profiles.to_parquet(Path(self.user_profiles_path, name), engine='fastparquet')

    def save_top_n_filter(self, top_n_selected: pd.DataFrame, n: int = 5, index: int = 0, save_dir: str = 'base', normalized: bool = False, filter_string: str = ""):
        base_path = Path(self.approximation_path, save_dir)
        self._create_path_if_not_exists(base_path)

        top_n_selected.to_parquet(Path(base_path, f"selected_top_{n}{'_normalized' if normalized else ''}{filter_string}_part_{index}.parquet"), engine='fastparquet')

    def save_scores(self, scores: pd.DataFrame, index: int = 0, model_dir: str = 'base'):
        base_path = Path(self.scores_path, model_dir)
        self._create_path_if_not_exists(base_path)

        scores.to_parquet(Path(base_path, f"score_part_{index}.parquet"), engine='fastparquet')

    def load_top_n_filter(self, n: int = 5, save_dir: str = 'base', normalized: bool = False, filter_string: str = ""):
        base_path = Path(self.approximation_path, save_dir)

        scores = pd.read_parquet(base_path.joinpath(Path(f"selected_top_{n}{'_normalized' if normalized else ''}{filter_string}_part_{0}.parquet")), engine='fastparquet')
        for index in range(1, self._amount_of_top_n_batches):
            to_add = pd.read_parquet(base_path.joinpath(Path(f"selected_top_{n}{'_normalized' if normalized else ''}{filter_string}_part_{index}.parquet")), engine='fastparquet')
            scores = pd.concat([scores, to_add], ignore_index=True)
        return scores

    def load_embeddings(self, total: int = None):
        if total is None or total > self._amount_of_embeddings_batches:
            total = self._amount_of_embeddings_batches

        embeddings = pd.read_parquet(Path(self.embeddings_path, f"embedding_part_{0}.parquet"), engine='fastparquet')
        for index in range(1, total):
            to_add = pd.read_parquet(Path(self.embeddings_path, f"embedding_part_{index}.parquet"), engine='fastparquet')
            embeddings = pd.concat([embeddings, to_add], ignore_index=True)

        return embeddings

    def load_business_profiles(self, name: str = "BASIC_BUSINESS_PROFILES.parquet"):
        return pd.read_parquet(Path(self.business_profile_path, name), engine='fastparquet')

    def load_user_profiles(self, name: str = "BASIC_USER_PROFILES.parquet") -> pd.DataFrame:
        return pd.read_parquet(Path(self.user_profiles_path, name), engine='fastparquet')

    def load_sentiment(self) -> pd.DataFrame:
        return self.load_scores(batches=10)[['review_id', 'label_sentiment', 'score_sentiment']]

    def load_scores(self, model_dir: str = 'base', batches: int = None) -> pd.DataFrame:
        if batches is None:
            batches = self._amount_of_scores_batches

        base_path = Path(self.scores_path, model_dir)
        self._create_path_if_not_exists(base_path)

        scores = pd.read_parquet(base_path.joinpath(Path(f"score_part_{0}.parquet")), engine='fastparquet')
        for index in range(1, batches):
            to_add = pd.read_parquet(base_path.joinpath(Path(f"score_part_{index}.parquet")), engine='fastparquet')
            scores = pd.concat([scores, to_add], ignore_index=True)
        return scores

    def load_approximation(self, model_dir: str = 'base') -> pd.DataFrame:
        base_path = Path(self.approximation_path, model_dir)

        approximations = pd.read_parquet(base_path.joinpath(Path(f"approximation_part_{0}.parquet")),
                                         engine='fastparquet').astype(np.float16)
        for index in range(1, self._amount_of_approximation_batches):
            to_add = pd.read_parquet(base_path.joinpath(Path(f"approximation_part_{index}.parquet")),
                                     engine='fastparquet').astype(np.float16)
            approximations = pd.concat([approximations, to_add], ignore_index=True)
        return approximations

    def load_zero_shot_classes(self) -> pd.DataFrame:
        scores = pd.read_parquet(Path(self.zero_shot_classes_path, f"zero_shot_classes_{0}.parquet"),
                                 engine='fastparquet')
        for index in range(1, self._amount_of_zero_shot_batches):
            to_add = pd.read_parquet(Path(self.zero_shot_classes_path, f"zero_shot_classes_{index}.parquet"),
                                     engine='fastparquet')
            scores = pd.concat([scores, to_add], ignore_index=True)
        return scores

    def is_available_embeddings(self):
        path = self.embeddings_path
        if not path.is_dir():
            path.mkdir()
        available_files = {file.name for file in os.scandir(path)}
        required_files = {f"embedding_part_{index}.parquet" for index in range(self._amount_of_embeddings_batches)}
        return required_files.issubset(available_files)

    def is_available_top_n(self, n: int = 5, model_dir: str = 'base', normalized: bool = False, filter_string: str = ""):
        path = self.approximation_path.joinpath(Path(model_dir))
        if not path.is_dir():
            path.mkdir()
        available_files = {file.name for file in os.scandir(path)}
        required_files = {f"selected_top_{n}{'_normalized' if normalized else ''}{filter_string}_part_{index}.parquet" for index in range(self._amount_of_top_n_batches)}
        return required_files.issubset(available_files)

    def is_available_scores(self, model_dir: str = 'base') -> bool:
        path = self.scores_path.joinpath(Path(model_dir))
        if not path.is_dir():
            path.mkdir()
        available_files = {file.name for file in os.scandir(path)}
        required_files = {f"score_part_{index}.parquet" for index in range(self._amount_of_scores_batches)}
        return required_files.issubset(available_files)

    def is_available_approximation(self, model_dir: str = 'base') -> bool:
        path = self.approximation_path.joinpath(Path(model_dir))
        if not path.is_dir():
            path.mkdir()
        available_files = {file.name for file in os.scandir(path)}
        required_files = {f"approximation_part_{index}.parquet" for index in range(self._amount_of_approximation_batches)}
        return required_files.issubset(available_files)

    def is_available_zero_shot_classes(self) -> bool:
        available_files = {file.name for file in os.scandir(self.zero_shot_classes_path)}
        required_files = {f"zero_shot_classes_{index}.parquet" for index in range(self._amount_of_zero_shot_batches)}
        return required_files.issubset(available_files)

    def is_available_sentiment(self) -> bool:
        return self.is_available_scores()
