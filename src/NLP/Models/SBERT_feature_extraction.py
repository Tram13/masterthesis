import pandas as pd
from sentence_transformers import SentenceTransformer


class SentenceBERT:
    # https://www.sbert.net/docs/pretrained_models.html
    def __init__(self, model_name: str = 'all-mpnet-base-v2') -> None:
        self.model = SentenceTransformer(model_name)

    def get_features(self, sentences: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(self.model.encode(sentences=sentences))


if __name__ == '__main__':
    # Our sentences we like to encode
    s = ['This framework generates embeddings for each input sentence',
         'Sentences are passed as a list of string.',
         'The quick brown fox jumps over the lazy dog.']
    # Model to use
    SBERTmodel = SentenceBERT()
    embedding = SBERTmodel.get_features(sentences=s)
    print(embedding)
    print(embedding.shape)