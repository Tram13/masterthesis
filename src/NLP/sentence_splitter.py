import logging

import pandas as pd
from pathlib import Path
from spacy.lang.en import English
from tqdm import tqdm

from src.tools.config_parser import ConfigParser

tqdm.pandas()


class SentenceSplitter:

    def __init__(self, verbose: bool = True) -> None:
        self.nlp = English()
        self.nlp.add_pipe('sentencizer')
        self.data_path = Path(ConfigParser().get_value('data', 'data_path'))
        self.cache_path = Path(self.data_path, ConfigParser().get_value('data', 'cache_directory'))
        self.cache_fname = 'splitted_reviews.parquet'
        self.verbose = verbose

    def _split_text_into_sentences(self, text: str) -> list[str]:
        return [sent.text for sent in self.nlp(text).sents]

    def _load_splitted_reviews_from_cache(self):
        logging.info('Reading splitted reviews from cache...')
        try:
            splitted_reviews = pd.read_parquet(Path(self.cache_path, self.cache_fname), engine='fastparquet')
        except OSError:
            logging.info('Could not read splitted reviews from cache, splitting them now...')
            splitted_reviews = None
        return splitted_reviews

    def _save_splitted_reviews_in_cache(self, splitted_reviews: pd.DataFrame):
        splitted_reviews.to_parquet(Path(self.cache_path, self.cache_fname), engine='fastparquet')

    def split_reviews(self, reviews: pd.Series, read_cache=True, save_in_cache=True):
        if read_cache:
            splitted_reviews = self._load_splitted_reviews_from_cache()
            if splitted_reviews is not None:
                return splitted_reviews

        if self.verbose:
            splitted_reviews = pd.DataFrame(reviews.progress_map(self._split_text_into_sentences))
        else:
            splitted_reviews = pd.DataFrame(reviews.map(self._split_text_into_sentences))
        # split sentences out in pd.dataframe while keeping indices of review
        splitted_reviews = splitted_reviews.explode('text').reset_index()
        splitted_reviews['text'] = splitted_reviews['text'].map(str.strip)

        if save_in_cache:
            self._save_splitted_reviews_in_cache(splitted_reviews)

        return splitted_reviews




