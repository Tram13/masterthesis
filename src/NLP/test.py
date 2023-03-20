from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.NLP.ListDataset import ListDataset
from src.NLP.zero_shot_classification import ZeroShotClassification
from src.data.data_reader import DataReader
from src.tools.config_parser import ConfigParser

classifier = ZeroShotClassification(classes=["food", "service", "environment"])

print('hello data')

# data_path = Path(ConfigParser().get_value('data', 'data_path'))
# cache_path = Path(data_path, ConfigParser().get_value('data', 'cache_directory'))
# cache_fname = 'splitted_reviews.parquet'
# splitted_reviews = pd.read_parquet(Path(cache_path, cache_fname), engine='fastparquet')

_, reviews, _ = DataReader().read_data()

print('hello gpu')

print(len(reviews))

text = list(reviews['text'].head(1280))
print(len(text))
list(tqdm(classifier.pipeline(ListDataset(text),
                              candidate_labels=["food", "service", "environment"],
                              multi_label=True,
                              batch_size=8,
                              truncation=True), total=len(text)))
