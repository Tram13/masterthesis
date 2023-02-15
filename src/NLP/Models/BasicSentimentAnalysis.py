from transformers import pipeline


class BasicSentimentAnalysis:

    def __init__(self) -> None:
        # default model for sentiment analysis is
        # 'distilbert-base-uncased-finetuned-sst-2-english'
        self.pipeline = pipeline("sentiment-analysis")

    def get_sentiment(self, text: list[str]) -> list[dict]:
        return self.pipeline(text)


if __name__ == '__main__':
    test_model = BasicSentimentAnalysis()

    t1 = ["Please read the analysis.", "You'll be amazed."]
    t2 = ["This restaurant sucks.", "It has the worst staff and terrible food.", "Only Jonny liked it."]

    output = test_model.get_sentiment(t1)
    print(output)

    output = test_model.get_sentiment(t2)
    print(output)

