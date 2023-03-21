import torch
from transformers import pipeline
from tqdm.auto import tqdm

from src.NLP.utils.ListDataset import ListDataset


# todo finetuning
class BasicSentimentAnalysis:

    def __init__(self, verbose: bool = True) -> None:
        # default model for sentiment analysis is
        # 'distilbert-base-uncased-finetuned-sst-2-english'
        self.pipeline = pipeline(task="sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english',
                                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),)
        self.verbose = verbose

    def get_sentiment(self, text: list[str]) -> list[dict]:
        if self.verbose:
            return list(tqdm(self.pipeline(ListDataset(text), batch_size=128, truncation=True), total=len(text)))
        else:
            return list(self.pipeline(ListDataset(text), batch_size=128, truncation=True))


if __name__ == '__main__':
    test_model = BasicSentimentAnalysis()

    t1 = ["Please read the analysis.", "You'll be amazed."]
    t2 = ["This restaurant sucks.", "It has the worst staff and terrible food.", "Only Jonny liked it."]

    output = test_model.get_sentiment(t1)
    print(output)

    output = test_model.get_sentiment(t2)
    print(output)
