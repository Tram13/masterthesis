import pandas as pd

from NLP.ModelsImplementations.BasicSentimentAnalysis import BasicSentimentAnalysis


# add sentiment label and score to the DataFrame
def sentiment_analysis_sentences(reviews: pd.DataFrame, verbose: bool = True):
    sentiment_analyzer = BasicSentimentAnalysis(verbose=verbose)
    df_sentiment = pd.DataFrame(sentiment_analyzer.get_sentiment(list(reviews['text'])))
    df_sentiment.columns = ['label_sentiment', 'score_sentiment']
    df_sentiment['label_sentiment'].replace(['NEGATIVE', 'POSITIVE'], [-1, 1], inplace=True)
    return pd.concat([reviews, df_sentiment], axis=1)
