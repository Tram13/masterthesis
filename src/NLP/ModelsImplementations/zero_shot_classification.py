import torch
from transformers import pipeline
from tqdm import tqdm

from NLP.utils.ListDataset import ListDataset


class ZeroShotClassification:

    def __init__(self, classes: list[str] = None, verbose: bool = True,
                 model: str = "Narsil/deberta-large-mnli-zero-cls") -> None:
        # "Narsil/deberta-large-mnli-zero-cls"
        model = "facebook/bart-large-mnli"
        self.pipeline = pipeline(
            task="zero-shot-classification",
            model=model,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.verbose = verbose
        self.classes = classes

    def get_class(self, text: list[str], multi_label: bool = True) -> list[dict]:
        if self.verbose:
            return list(tqdm(self.pipeline(ListDataset(text),
                                           candidate_labels=self.classes,
                                           multi_label=multi_label,
                                           batch_size=8,
                                           truncation=True), total=len(text)))
        else:
            return list(self.pipeline(ListDataset(text),
                                      candidate_labels=self.classes,
                                      multi_label=multi_label,
                                      batch_size=8,
                                      truncation=True))
