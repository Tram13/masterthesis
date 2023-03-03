import pandas as pd
from spacy.lang.en import English
from tqdm import tqdm
tqdm.pandas()


class SentenceSplitter:

    def __init__(self) -> None:
        self.nlp = English()
        self.nlp.add_pipe('sentencizer')

    def split_text_into_sentences(self, text: str) -> list[str]:
        return [sent.text for sent in self.nlp(text).sents]


def split_reviews(reviews: pd.Series):
    splitter = SentenceSplitter()
    splitted_reviews = pd.DataFrame(reviews.progress_map(splitter.split_text_into_sentences))
    # split sentences out in pd.dataframe while keeping indices of review
    splitted_reviews = splitted_reviews.explode('text').reset_index()
    splitted_reviews['text'] = splitted_reviews['text'].map(str.strip)
    return splitted_reviews
