import pandas as pd

from src.NLP.Models.BasicSentimentAnalysis import BasicSentimentAnalysis


# add sentiment label and score to the DataFrame
def sentiment_analysis_sentences(reviews: pd.DataFrame):
    sentiment_analyzer = BasicSentimentAnalysis()
    df_sentiment = sentiment_analyzer.get_sentiment(list(reviews['text']))
    df_sentiment.columns = ['label_sentiment', 'score_sentiment']
    return pd.concat([reviews, df_sentiment], axis=1)