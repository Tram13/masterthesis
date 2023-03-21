from pathlib import Path
from bertopic import BERTopic
from src.tools.config_parser import ConfigParser


class NLPModels:

    def __init__(self):
        self.model_path = Path(ConfigParser().get_value('model', 'model_dir'))

        if not self.model_path.is_dir():
            self.model_path.mkdir()

        self.current_bert_model_name = self.model_path.joinpath(
            Path(ConfigParser().get_value('model', 'current_bert_model')))
        self.save_bert_model_path = self.model_path.joinpath(
            Path(ConfigParser().get_value('model', 'save_bert_model_name')))

    def load_model(self, name: str = None) -> BERTopic:
        if name is None:
            return BERTopic.load(self.current_bert_model_name)

        return BERTopic.load(self.model_path.joinpath(Path(name)))

    def save_model(self, model: BERTopic, name: str = None):
        if name is None:
            return model.save(self.save_bert_model_path)

        return model.save(self.model_path.joinpath(Path(name)))
