# Implementation
Deze directory bevat alle code voor de implementaties van alle geteste modellen.
## MultiLayerPerceptronPredictor
Deze klasse bevat de implementatie van het neuraal netwerk. De verschillende geteste architecturen (zie HS4 en HS5) worden voorgesteld in `multilayer_perceptron1` tot `multilayer_perceptron8`.
De superklasse `MultiLayerPerceptronPredictor` bevat alle code om modellen op te slaan, en de specifieke architecturen staan beschreven in de `__init__()`-functie van de subklassen.
## bias
De implementatie van de Bias recommender, gebaseerd op de formule van de Lenskit recommender.
## cb
De implementatie van de Content-Based Filtering recommender, gebaseerd op de implementatie gebruikt uit een practicum van het vak 'Aanbevelingssystemen'.
## cf
De implementaties van de IICF en UUCF recommender, gebaseerd op de implementatie gebruikt uit een practicum van het vak 'Aanbevelingssystemen'.
## deepconn
De implementatie gebruikt om de DeepCoNN RMSE te bepalen.
## rf
De implementatie van Random Forest.
## wide_and_deep
De implementatie gebruikt om de Wide & Deep Learning RMSE te bepalen.
