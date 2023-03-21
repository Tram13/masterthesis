from pathlib import Path
from src.tools.config_parser import ConfigParser
import pandas as pd


class NLPCache:

    def __init__(self, amount_of_bert_batches: int = 10, amount_of_zero_shot_batches: int = 30):
        self.cache_path = Path(ConfigParser().get_value('cache', 'nlp_cache_dir'))
        self.user_profiles_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'user_profiles_dir')))
        self.bert_scores_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'scores_dir')))
        self.sentiment_path = self.bert_scores_path
        self.zero_shot_classes_path = self.cache_path.joinpath(Path(ConfigParser().get_value('cache', 'zero_shot_dir')))

        self._make_dirs()

        # config amount of batches that was done
        self._amount_of_bert_batches = amount_of_bert_batches
        self._amount_of_zero_shot_batches = amount_of_zero_shot_batches

    def _make_dirs(self):
        if not self.cache_path.is_dir():
            self.cache_path.mkdir()

        if not self.user_profiles_path.is_dir():
            self.user_profiles_path.mkdir()

        if not self.bert_scores_path.is_dir():
            self.bert_scores_path.mkdir()

        if not self.sentiment_path.is_dir():
            self.sentiment_path.mkdir()

        if not self.zero_shot_classes_path.is_dir():
            self.zero_shot_classes_path.mkdir()

    def load_user_profiles(self, name: str = "BASIC_USER_PROFILES.parquet") -> pd.DataFrame:
        return pd.read_parquet(Path(self.user_profiles_path, name), engine='fastparquet')

    def load_sentiment(self) -> pd.DataFrame:
        return self.load_bert_scores()[['review_id', 'label_sentiment', 'score_sentiment']]

    def load_bert_scores(self) -> pd.DataFrame:
        scores = pd.read_parquet(Path(self.bert_scores_path, f"score_part_{0}.parquet"), engine='fastparquet')
        for index in range(1, self._amount_of_bert_batches):
            to_add = pd.read_parquet(Path(self.bert_scores_path, f"score_part_{index}.parquet"), engine='fastparquet')
            scores = pd.concat([scores, to_add])
        return scores

    def load_zero_shot_classes(self) -> pd.DataFrame:
        scores = pd.read_parquet(Path(self.zero_shot_classes_path, f"zero_shot_classes_{0}.parquet"), engine='fastparquet')
        for index in range(1, self._amount_of_zero_shot_batches):
            to_add = pd.read_parquet(Path(self.zero_shot_classes_path, f"zero_shot_classes_{index}.parquet"), engine='fastparquet')
            scores = pd.concat([scores, to_add])
        return scores
