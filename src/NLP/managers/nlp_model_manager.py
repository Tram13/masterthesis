from pathlib import Path
from bertopic import BERTopic
from tools.config_parser import ConfigParser


class NLPModels:

    def __init__(self):
        self.model_path = Path(ConfigParser().get_value('model', 'model_dir'))

        if not self.model_path.is_dir():
            self.model_path.mkdir()

        self.save_bert_model_path = self.model_path.joinpath(
            Path(ConfigParser().get_value('model', 'save_bert_model_name')))

        self.models, self.current_bert_model_name = self._generate_model_dict()

    @staticmethod
    def _generate_model_dict():
        amount_of_models = int(ConfigParser().get_value('available_models', 'amount'))
        default_model_index = int(ConfigParser().get_value('model', 'current_bert_model_index'))
        model_names = []
        model_dirs = []
        for index in range(amount_of_models):
            model_names.append(ConfigParser().get_value('available_models', f'model_{index}'))
            model_dirs.append(ConfigParser().get_value('available_models', f'directory_{index}'))

        model_dir_dict = {name: directory for name, directory in zip(model_names, model_dirs)}
        return model_dir_dict, model_names[default_model_index]

    def load_model(self, name: str = None) -> BERTopic:
        if name is None:
            return BERTopic.load(Path(self.model_path, self.current_bert_model_name))

        return BERTopic.load(self.model_path.joinpath(Path(name)))

    def save_model(self, model: BERTopic, name: str = None):
        if name is None:
            return model.save(self.save_bert_model_path)

        return model.save(self.model_path.joinpath(Path(name)))

    def get_dir_for_model(self, model_name):
        return self.models[model_name]


