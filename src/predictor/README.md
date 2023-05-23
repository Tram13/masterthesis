# Predictor
Deze directory bevat alle code en modellen met betrekking tot het aanbevelingssysteem-gedeelte, waarbij de inputvectoren reeds gemaakt zijn en enkel nog verwerkt worden, door bijvoorbeeld een neuraal netwerk.
## Bias Model Validation
Deze notebook geeft de code om de Bias recommender uit te voeren en de performantie te valideren.
## LossScore 
Deze klasse bevat de aangepaste lossfuncties zoals beschreven in hoofdstuk 5. Om deze te gebruiken moet lijn 104 in `NeuralNetworkTrainer` aangepast worden.
## NeuralNetworkTrainer
Deze klasse omvat alle functionaliteit die nodig is om een `MultiLayerPerceptronPredictor`-model te trainen en te valideren.
Zie onderstaand voorbeeld om deze klasse te gebruiken: 
```python
from data.data_reader import DataReader
from tools.user_profiles_manager import UserProfilesManager
from tools.restaurant_profiles_manager import RestaurantProfilesManager
from predictor.neural_network_trainer import NeuralNetworkTrainer


train_test_data = DataReader().read_data()
up_params = UserProfilesManager().get_best()
rp_params = RestaurantProfilesManager().get_best()

nn_trainer = NeuralNetworkTrainer(up_params, rp_params, *train_test_data)
# Gebruik nu de trainer met een MultiLayerPerceptronPredictor en een torch optimizer (zoals Adagrad)
nn_trainer.train(model, optimizer)
```
Of zie de functie `main_train_models_with_same_data_splits()` uit `main.py`. 
## Trained MLP validation
Deze notebook omvat code die gebruikt werd in hoofdstuk 5 om de analyses uit te voeren. Doordat we in dat hoofdstuk veel experimenten doen met alternatieve implementaties, zal er soms manueel iets moeten worden aangepast in deze notebook. Bijvoorbeeld: de in te laden optimizerfunctie, het aantal in te laden modellen, etc.
Deze notebook verwacht dat alle modellen die gevalideerd moeten worden zich bevinden in `src.predictor.Models.mlp`. Het voert een test uit op de testdata, met niet vooraf gebruikte gebruikers- en restaurantprofielen.

Daarna worden grafieken en histogrammen gemaakt over één van deze ingeladen modellen, afhankelijk van de parameter `index` in de notebook.
Afhankelijk van de gewenste analyse zal er eventueel hier en daar wat code moeten aangepast worden. Zie ook de klasse `tools.ModelLoader` om te helpen bij het implementeren.

## Trained RF validation
Deze notebook geeft de code om de Random Forest recommender uit te voeren en de performantie te valideren.
