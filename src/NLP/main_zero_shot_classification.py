import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.NLP.df_NLP_manipulation.df_zero_shot_class import zero_shot_class
from src.NLP.managers.nlp_cache_manager import NLPCache


def main_calculate_zero_shot_classification(reviews: pd.Series, classes: list[str] = None, amount_of_batches: int = 30,
                                            use_cache: bool = True):
    nlp_cache = NLPCache()

    if classes is None:
        classes = ["food", "service", "environment"]

    if not use_cache or not nlp_cache.is_available_zero_shot_classes():
        logging.warning(
            f'Cache is not being used: allowed: {use_cache} - available: {nlp_cache.is_available_zero_shot_classes()}')
        logging.info('Calculating zero shot classification scores...')
        for index, batch in enumerate(tqdm(np.array_split(reviews, amount_of_batches), desc="Scores ZSC")):
            print()
            zero_shot_features = zero_shot_class(batch, classes=classes)
            zero_shot_features.columns = [str(x) for x in zero_shot_features.columns]
            zero_shot_features.to_parquet(
                nlp_cache.zero_shot_classes_path.joinpath(Path(f"zero_shot_classes_{index}.parquet")),
                engine='fastparquet'
            )

    logging.info('Completed calculation, loading in data to return...')
    return nlp_cache.load_zero_shot_classes()
