import pandas as pd
from src.NLP.zero_shot_classification import ZeroShotClassification


# add sentiment label and score to the DataFrame
def zero_shot_class(reviews: pd.Series, classes: list[str] = None, verbose: bool = True):
    if classes is None:
        classes = ["food", "service", "hygiene", "good", "bad"]
    zero_shot = ZeroShotClassification(verbose=verbose, classes=classes)
    df_zero_shot = pd.DataFrame(zero_shot.get_class(list(reviews)))
    # df_zero_shot.columns = ['label_sentiment', 'score_sentiment']
    return df_zero_shot
