from pathlib import Path
from tools.config_parser import ConfigParser
import pandas as pd
import os


class NLPCache:

    def __init__(self, amount_of_scores_batches: int = 10, amount_of_zero_shot_batches: int = 30):
        self.cache_path = Path(ConfigParser().get_value('cache', 'nlp_cache_dir'))
        self.user_profiles_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'user_profiles_dir')))
        self.scores_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'scores_dir')))
        self.sentiment_path = self.scores_path
        self.zero_shot_classes_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'zero_shot_dir')))

        self._make_dirs()

        # config amount of batches that was done
        self._amount_of_scores_batches = amount_of_scores_batches
        self._amount_of_zero_shot_batches = amount_of_zero_shot_batches

    def _make_dirs(self):
        if not self.cache_path.is_dir():
            self.cache_path.mkdir()

        if not self.user_profiles_path.is_dir():
            self.user_profiles_path.mkdir()

        if not self.scores_path.is_dir():
            self.scores_path.mkdir()

        if not self.sentiment_path.is_dir():
            self.sentiment_path.mkdir()

        if not self.zero_shot_classes_path.is_dir():
            self.zero_shot_classes_path.mkdir()

    def load_user_profiles(self, name: str = "BASIC_USER_PROFILES.parquet") -> pd.DataFrame:
        return pd.read_parquet(Path(self.user_profiles_path, name), engine='fastparquet')

    def load_sentiment(self) -> pd.DataFrame:
        return self.load_scores()[['review_id', 'label_sentiment', 'score_sentiment']]

    def load_scores(self) -> pd.DataFrame:
        scores = pd.read_parquet(Path(self.scores_path, f"score_part_{0}.parquet"), engine='fastparquet')
        for index in range(1, self._amount_of_scores_batches):
            to_add = pd.read_parquet(Path(self.scores_path, f"score_part_{index}.parquet"), engine='fastparquet')
            scores = pd.concat([scores, to_add], ignore_index=True)
        return scores

    def load_zero_shot_classes(self) -> pd.DataFrame:
        scores = pd.read_parquet(Path(self.zero_shot_classes_path, f"zero_shot_classes_{0}.parquet"),
                                 engine='fastparquet')
        for index in range(1, self._amount_of_zero_shot_batches):
            to_add = pd.read_parquet(Path(self.zero_shot_classes_path, f"zero_shot_classes_{index}.parquet"),
                                     engine='fastparquet')
            scores = pd.concat([scores, to_add], ignore_index=True)
        return scores

    def is_available_scores(self) -> bool:
        available_files = {file.name for file in os.scandir(self.scores_path)}
        required_files = {f"score_part_{index}.parquet" for index in range(self._amount_of_scores_batches)}
        return required_files.issubset(available_files)

    def is_available_zero_shot_classes(self) -> bool:
        available_files = {file.name for file in os.scandir(self.zero_shot_classes_path)}
        required_files = {f"zero_shot_classes_{index}.parquet" for index in range(self._amount_of_zero_shot_batches)}
        return required_files.issubset(available_files)

    def is_available_sentiment(self) -> bool:
        return self.is_available_scores()
