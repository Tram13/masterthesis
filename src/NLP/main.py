import pandas as pd

from src.NLP.Models.SBERT_feature_extraction import SentenceBERT
from src.NLP.df_NLP_manipulation.df_sentiment_analysis import sentiment_analysis_sentences
from src.NLP.dimensionality_reduction import DimensionalityReduction
from src.NLP.sentence_splitter import SentenceSplitter
from src.NLP.clustering import ClusteringAlgorithms






def run_main(reviews: pd.Series):
    splitter = SentenceSplitter()
    splitted_reviews = pd.DataFrame(reviews.map(splitter.split_text_into_sentences))
    # split sentences out in pd.dataframe while keeping indices of review
    splitted_reviews = splitted_reviews.explode('text').reset_index()
    splitted_reviews['text'] = splitted_reviews['text'].map(str.strip)

    # clustering label
    reviews = cluster_sentences(splitted_reviews)

    # sentiment label+score for each sentence
    reviews = sentiment_analysis_sentences(reviews)


if __name__ == '__main__':
    reviews_input = pd.read_csv('tmp.pd')['text']
    x = run_main(reviews_input)
    print(x)

