from spacy.lang.en import English


class SentenceSplitter:

    def __init__(self) -> None:
        self.nlp = English()
        self.nlp.add_pipe('sentencizer')

    def split_text_into_sentences(self, text: str) -> list[str]:
        return [sent.text for sent in self.nlp(text).sents]


if __name__ == '__main__':
    splitter = SentenceSplitter()
    t1 = "Please read the analysis. You'll be amazed."
    t2 = "This restaurant sucks. It has the worst staff and terrible food. Only Jonny liked it."
    s = splitter.split_text_into_sentences(t1)
    print(s)
    s = splitter.split_text_into_sentences(t2)
    print(s)

