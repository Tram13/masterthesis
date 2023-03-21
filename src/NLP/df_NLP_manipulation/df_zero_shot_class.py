import pandas as pd
from src.NLP.zero_shot_classification import ZeroShotClassification


# add sentiment label and score to the DataFrame
def zero_shot_class(reviews: pd.Series, classes: list[str] = None, verbose: bool = True):
    if classes is None:
        classes = ["food", "service", "environment"]
    zero_shot = ZeroShotClassification(verbose=verbose, classes=classes)
    classifications = zero_shot.get_class(list(reviews))
    df_zero_shot = pd.DataFrame(
        [{lab: score for lab, score in zip(test_code['labels'], test_code['scores'])} for test_code in classifications]
    )
    return df_zero_shot
