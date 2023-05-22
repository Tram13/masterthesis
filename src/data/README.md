# Data Parsing
Deze directory bevat 2 klassen: `DataReader` en `DataPreparer`.
## DataReader
De functionaliteit van deze klasse beperkt zich tot het inlezen van de originele (onverwerkte) Yelp Dataset.
Hier gaan we dus data inlezen, de train-test split maken en in kleine mate feature engineering toepassen.


De code om de restaurants, users en reviews te parsen staat ook in `src/data/data analysis/*.ipynb` Het is makkelijker om de code daar te volgen dan in deze klasse. Er staat hier en daar ook uitleg bij de beslissingen die we namen.
```python
from data.data_reader import DataReader

# Gebruik van deze klasse
train_data, test_data = DataReader().read_data()
```

## DataPreparer
Deze klasse omvat de functionaliteit om de data van `DataReader` te transformeren naar input die geldig is voor het neuraal netwerk.

`parse_data_train_test()` combineert de aparte Users, Reviews, Restaurants DataFrames naar één DataFrame a.d.h.v. joins
Daarna wordt bij zowel de train- als testset een deel van de data gebruikt als generation set voor de profielen zoals  gespecifieerd in de parameter `profile_params`.
Deze profielen worden dan toegevoegd aan de overige data en teruggegeven.

```python
from data.data_preparer import DataPreparer

# Gebruik van deze klasse, voor de ingeladen train- en testdata uit DataReader
# De profile_params stellen voor welke gebruikers- en restaurantprofielen moeten gebruikt worden, zie `tools.user_profiles_manager` en `tools.restaurant_profiles_manager`
training_input, test_input, training_output, test_output = DataPreparer.parse_data_train_test(train_data, test_data, profile_params)
```