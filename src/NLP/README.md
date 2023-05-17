# BERTopic model trainen

Voor een offline model (niet aanbevolen om offline modellen te trainen) kan dit via de volgende functie uit `src/NLP/main_offline_BERTopic.py`:

- `reviews`: Pandas Series met de tekstuele reviews. Deze mag niet te groot zijn (afhankelijk van de beschikbare hoeveelheid RAM).
- `embeddings` Het is mogelijk om precomputed of custom embeddings mee te geven voor gebruik. Aanbevolen is om ze te laten berekenen, met andere woorden `embbedings=None`
- `do_precompute_and_save_embeddings`: Dit zal zorgen dat we embeddings precomputen indien deze niet zijn meegegeven. Om deze ook op te slaan moet de parameter `save_path` ingevuld worden.
- `save_path`: Locatie waar de precommuted embeddings worden opgeslagen (indien ze geprecompute worden).
- `model_name`: De bestandsnaam van het model, gebruikt om op te slaan.
- `calculate_profiles`: Indien na het trainen van het model ook de profielen moeten berekend worden (niet aanbevolen om dit via deze methode te doen).
- `use_sentiment`: Indien sentiment analysis in de profielen gebruikt moet worden (niet aanbevolen om dit via deze methode te doen).
- `generate_prediction_data`: Het gebruik van HDBSCAN zal een prediction matrix aanmaken, dit zal voor ongeziene data zijn. Indien het model geen ongeziene data moet clusteren is het mogelijk om deze parameter op `False` te zetten.

```python
def main_BERTopic(reviews: pd.Series,
                  embeddings: np.ndarray = None,
                  do_precompute_and_save_embeddings: bool = False,
                  save_path: Path = None, 
                  use_sentiment: bool = False,
                  model_name: str = "offline_test_model",
                  calculate_profiles: bool = False,
                  generate_prediction_data: bool = True):
```

Dit een online model (aanbevolen) kan via de volgende functie uit `src/NLP/main_online_BERTopic.py`: 

- `reviews`: Pandas Series met de tekstuele reviews
- `sentence_batch_size`: Het online model fit in meerdere batches, elke batch zal ongeveer `sentence_batch_size` zinnen bevatten.
- `model_name`: De bestandsnaam van het model, gebruikt om op te slaan.
- `dim_red_components`: Het online model maakt gebruik van Incremental PCA, deze zal de input features van 786 omzetten naar `dim_red_components` features.
- `max_topics`: Het model zal maximaal `max_topics` genereren, indien de dataset klein of `max_topics` groot is kan dit minder zijn.
- `guided_topics`: Lijst van lijsten waarbij elke lijst een voorafgedefinieërde topics voorstelt. De gebruikte topics in de thesis staan vermeld in `src/NLP/guided_topics.txt`, deze is identiek aan de versie in de cache `src/NLP/cache/guided_topics/NLP_categories.txt`

```python
def create_model_online_BERTopic(reviews: pd.Series,
                                 sentence_batch_size: int = 500_000,
                                 model_name: str = None,
                                 dim_red_components: int = 15,
                                 max_topics: int = 200,
                                 guided_topics: list[list[str]] = None):
```

# Evalueren van een clustering

We kunnen clusteringsmetrieken uitrekenen voor de verschillende BERTopic modellen. Dit kan aan de hand van de volgende functie uit `src/NLP/utils/evaluate_model.py`. Deze methode zal respectievelijk de `calinski_harabasz_score`, `davies_bouldin_index` en `silhoutte_score` uitrekenen en vervolgens op een nieuwe lijn gescheiden door een komma toevoegen aan `src/metrics.csv`.

- `sentences`: De zinnen die gebruikt moeten worden voor (Let op: deze worden enkel gebruikt indien de cache geen embeddings bevat!)
- `model_name`: Naam van het model waarvan we de clustering moeten evalueren.
- `percentage`: Getal tussen 1-100, we gebruiken `percentage`% van de zinnen (tenzij `divide_10=True`)
- `divide_10`: Indien deze parameter op `True` staat zullen we in plaats van `percentage`% van de zinnen `percentage/10`% gebruiken 
- `dim_reduction`: Indien dimensionality reductie moet toegepast worden op embeddings, aangezien dit ook gebruikt voor het clustering is het aanbevolen dit altijd op `True` te zetten.

```python
def evaluate_model(sentences: pd.DataFrame, model_name: str, percentage: int, divide_10: bool = False, dim_reduction: bool = True):
```

Een voorbeeld script om het model met de naam `online_model_50top_85.bert` te evalueren is hieronder te vinden.

```python
import logging
from NLP.utils.sentence_splitter import SentenceSplitter
from NLP.utils.evaluate_model import evaluate_model


logging.basicConfig(level=logging.INFO)

# for this example we load all the splitted sentences from the cache
sentences = SentenceSplitter()._load_splitted_reviews_from_cache()
model_name = "online_model_50top_85.bert"

logging.info('Finished reading in data, starting evaluation...')

# using 0.1% of the data
logging.info("0.1% of the data")
evaluate_model(sentences, model_name, 1, True)

# using 0.5% of the data
logging.info("0.5% of the data")
evaluate_model(sentences, model_name, 5, True)

# using 1% of the data
logging.info("1% of the data")
evaluate_model(sentences, model_name, 1, False)

# using 2% of the data
logging.info("2% of the data")
evaluate_model(sentences, model_name, 2, False)
```
# Gebruikers- en restaurantprofielen genereren

Het is mogelijk om de functies in `src/NLP/main_user_profiles.py` te gebruiken, maar dit is niet aanbevolen. De makkelijkste manier is via de `ProfileCreator` in `src/NLP/profile_creator.py`. De werking hiervan wordt hieronder beschreven:

De eerste stap is een object van het type `ProfileCreator` aanmaken, deze bevat de volgende parameters:

- `model_name`: De naam van het model dat gebruik moet worden om profielen te generen.
- `use_sentiment_in_scores`: Boolean of er sentiment moet toegevoegd worden in de profielen.
- `approx_mode`: Indien `True` worden de profielen gegenereerd op basis van de representaties van het model. Indien `False` gebeurt dit via de clustering van het model.
- `approx_normalization`: Enkel relevant indien `approx_mode=True`: Na het selecteren van de top n meest relevante topics per zin worden deze waarden nog genormaliseerd indien deze parameter op `True` staat (aanbevolen: altijd `True`).
- `approx_amount_top_n`: Uit de approximatie van de relevante topics zullen enkel de ´approx_amount_top_n´ beste topics overgehouden worden.
- `filter_useful_topics`: Boolean of de topics gefiltered moeten worden. Let op: dit is een manuele filter op de topics die enkel geldig is op het model met de naam `online_model_400top_97.bert`.

Na het aanmaken van een object kunnen we de volgende methoden gebruiken om respectievelijk een gebruikersprofiel en restaurantsprofiel op te vragen. De enige parameter die nog ingevuld moet worden is `reviews`. Dit is een Pandas Dataframe met de reviews de gebruikt moeten worden om de profielen te genereren.

```python
def get_user_profile(self, reviews: pd.DataFrame):
def get_restaurant_profile(self, reviews: pd.DataFrame):
```

Andere relevante methoden is om een dictionary van de parameters op te vragen of een profile_creator inladen aan de hand van deze dictionary.
```python
def get_build_parameters(self) -> dict:
def load_from_dict(params: dict):   # STATIC METHOD
```

Een voorbeeld is hieronder te vinden, merk op dat we hier voor zowel gebruikers als restaurantprofielen geen sentiment gebruiken. Zoals beschreven in de thesis is het beter om dit wel te doen voor restaurantprofielen. Om dit op te lossen is het mogelijk om simpeleweg een andere instantie van de klasse `ProfileCreator` aan te maken die wel de juiste parameters heeft.

```python
from NLP.profiles_creator import ProfileCreator
from data.data_reader import DataReader


# inladen van volledige dataset
(_, reviews, _), _ = DataReader().read_data(no_train_test=True)

# Aanmaken ProfileCreator
profile_generator = ProfileCreator(
    model_name="online_model_50top_85.bert",    # gebruikt model
    use_sentiment_in_scores=False,              # geen sentiment gebruiken in de profielen
    approx_mode=True,                           # we gaan gebruik maken van de representatie van het model
    approx_normalization=True,                  # normalisatie staat aan (aanbevolen)
    approx_amount_top_n=5,                      # we zullen de 5 meeste relevante topîcs per review overhouden
    filter_useful_topics=False                  # Aangezien dit niet `online_model_400top_97.bert` is, is het niet mogelijk om te filteren
)

# uitlezen van de parameters, verzameld in een dictionary
parameter_dictionary = profile_generator.get_build_parameters()

# deze is identiek aan de profile_generator hierboven. Het maakt gebruik van dezelfde parameters.
the_same_profile_generator = ProfileCreator.load_from_dict(parameter_dictionary)

# Opvragen van alle gebruikers- en restaurantprofielen op basis van alle data (pd.DataFrame). 
# Indien een deel van de data gebruikt moet worden kan een subset of andere Dataframe met dezelfde kolomnamen meegegeven worden.
user_profiles = profile_generator.get_user_profile(reviews)
restaurant_profiles = profile_generator.get_restaurant_profile(reviews)
```

# Opsplitsen van zinnen
Aangezien de BERTopic modellen niet de volledige reviews als input nemen, maar deze opsplitsen in zinnen is er code om deze op te splitsen. Dit zal gebeuren aan de hand van de klasse `SentenceSplitter` te vinden in `src/NLP/utils/sentence_splitter.py`. Deze klasse neemt één boolean parameter namelijk `verbose`, indien `verbose=True` zal er een progressbar getoond worden terwijl de zinnen gesplitst worden.
Deze splitsing hoeft slechts eenmaal te gebeuren en duur voor de volledige dataset ongeveer 22 minuten. Na het splitsen wordt deze lokaal opgeslagen. Merk op dat deze cache altijd alle data zal bevatten, indien we een kleiner deel moet gesplitst worden kan dit opnieuwe gedaan worden of deels uitgelezen.

Na het aanmaken van een instantie van de klasse `SentenceSplitter` gebruiken we de volgende methode om de reviews op te splitsen:

- `reviews`: Een Pandas Series waarbij de kolom die de tekstuele reviews bevat `'text'` noemt. 
- `read_cache`: Indien mogelijk, laad de volledige dataset van opgesplitste reviews in van disk.  
- `save_in_cache`: Indien we niet van disk lezen zullen we de opgesplitste zinnen opslaan en indien ze al bestaan overschrijven.

```python
def split_reviews(self, reviews: pd.Series, read_cache: bool = True, save_in_cache: bool = True):
```

Een voorbeeld hoe we de volledige dataset opsplitsen in zinnen, indien mogelijk inlezen van de cache anders uitrekenen en opslaan.

```python
from data.data_reader import DataReader
from NLP.utils.sentence_splitter import SentenceSplitter


# Inladen van alle data (om te splitsen)
(_, reviews, _), _ = DataReader().read_data(no_train_test=True)

# aanmaken van een SentenceSplitter object met progressbar
sentence_splitter = SentenceSplitter(verbose=True)

# Tekstuele deel van reviews opsplitsen in zinnen (inlezen van cache indien deze bestaan), indien deze niet bestaan mogen ze opgeslagen worden
splitted_reviews = sentence_splitter.split_reviews(reviews['text'], read_cache=True, save_in_cache=True)
```


# Gebruik van caches (disk)

De code voor het opslaan en inlezen van de cache is te vinden in `src/NLP/managers/nlp_cache_manager.py`. Deze cache kunnen we gebruiken voor de volgende elementen:

TODO uitleggen hoe we aanmaken + batches parameters

- Inlezen van voorafgedefinieërde topics met de naam `name`
```python
def read_guided_topics(self, name: str = "NLP_categories.txt"):
```
- Inlezen, opslaan en controleren of ze bestaan van embeddings
```python
    def save_embeddings(self, embeddings, index):   # opslaan op bepaalde index
    def load_embeddings(self, total: int = None):   # inladen van de eerste N embeddings
    def is_available_embeddings(self):              # True als alle embeddings van 0 tot amount_of_embeddings_batches beschikbaar zijn
```

- Inlezen en opslaan van gebruikers- en restaurantprofielen op basis van naam
```python
  def save_user_profiles(self, user_profiles: pd.DataFrame, name: str = "BASIC_USER_PROFILES.parquet"):
  def load_user_profiles(self, name: str = "BASIC_USER_PROFILES.parquet"):

  def save_business_profiles(self, business_profiles: pd.DataFrame, name: str = "BASIC_BUSINESS_PROFILES.parquet"):
  def load_business_profiles(self, name: str = "BASIC_BUSINESS_PROFILES.parquet"):
```

- 

todo structuur
```

```

# Inlezen van een getraind BERTopic model
todo

# Overige informatie

- De implementatie van formules om gebruikers- en restaurantprofielen te genereren staan in `src/NLP/utils/scoring_functions.py`
- todo zero shot classification
- modellen in models folder
- helperklassen in ModelsImplementations
- configfile?