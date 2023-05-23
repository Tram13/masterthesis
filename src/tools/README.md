# Tools
Deze directory bevat helperfunctions die veelgebruikte functionaliteit implementeren
## ConfigParser
Deze klasse zal de bijgeleverde `config.ini` file parsen die in de root van het project staat. Dit is in essentie een wrapper voor de built-in `configparser`.
## ModelLoader
Deze klasse helpt om een `MultiLayerPerceptronPredictor`-model in te lezen.
```python
from tools.model_loader import ModelLoader
from predictor.implementations.multilayer_perceptron import MultiLayerPerceptronPredictor

model: MultiLayerPerceptronPredictor = ModelLoader.load_mlp_model("path_to_model.pt")
```
## PrintSilencer
Wordt momenteel niet gebruikt
## RestaurantProfilesManager en UserProfilesManager
Deze klassen bevatten de parameterconfiguraties van respectievelijk NLP-restaurantprofielen en NLP-gebruikersprofielen.
Ze kunnen gebruikt worden als `iterable` om alle mogelijke implemenataties te overlopen of om de beste parameters op te vragen.
Deze parameterconfiguraties kunnen gebruikt worden bij `DataPreparer.parse_data_train_test()` in de `profile_params`-parameter, of rechtstreeks om het overeenkomstig profiel te genereren:


```python
from tools.restaurant_profiles_manager import RestaurantProfilesManager
# Als iterator
for profile_params in RestaurantProfilesManager():
    pass

# Om het profiel te genereren dat overeenkomt met de parameterconfiguratie
# Rechtstreeks
from NLP.profiles_creator import ProfileCreator
best_config_NLP_restaurant_profiles = RestaurantProfilesManager().get_best()
profile = ProfileCreator.load_from_dict(best_config_NLP_restaurant_profiles).get_restaurant_profile(reviews_generation)
```
(Analoog voor `UserProfilesManager`)
## RestaurantReviewsDataset
Wrapper-klasse die de torch `Dataset`-klasse implementeert, specifiek voor de structuur van de Yelp-dataset.