from pathlib import Path
from transformers import BertTokenizer, TFBertModel, pipeline
from typing import Union


class ModelBERT:
    TOKENIZER_SUFFIX = ".TOK_BERT"
    MODEL_SUFFIX = ".MOD_BERT"

    def __init__(self, location: Union[Path | str] = 'bert-base-uncased', use_case: str = "feature-extraction") -> None:
        self.task = use_case
        self.tokenizer = BertTokenizer.from_pretrained(location)
        self.model = TFBertModel.from_pretrained(location)
        self.pipeline = pipeline(task=self.task, model=self.model, tokenizer=self.tokenizer)

    def save(self, save_location: Path) -> None:
        self.tokenizer.save_pretrained(save_location.with_suffix(self.TOKENIZER_SUFFIX))
        self.model.save_pretrained(save_location.with_suffix(self.MODEL_SUFFIX))

    def load(self, location: Union[Path | str]) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(location)
        self.model = TFBertModel.from_pretrained(location)
        self.pipeline = pipeline(task=self.task, model=self.model, tokenizer=self.tokenizer)

    def process_text(self, text):
        return self.pipeline(text, return_tensors=False, framework='tf')


if __name__ == '__main__':
    testmodel = ModelBERT()
    output = testmodel.process_text(['This is a sentence based on machine learning!', 'Working is very hard.'])
    print(output)
    print(type(output))
    print(len(output))
