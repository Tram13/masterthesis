TODO
Onderzoektips checken voor het gebruik van data - gdrp
Groeperen van maaltijden/keukens gebeurt manueel. Zou een model voor kunnen getrained worden, maar didn't bother. Probleem met gewone modellen is dat die te weinig onderscheid maken tussen verschillende eten-dingen, en niet kijken naar bvb afkomst of ingedrienten;
verschillende testen doen, met en zonder 'restaurantprofiel', waarbij we kijken hoe goed de performance omhoog gaat door restaurantprofielen ook aan te maken met tekstuele data (zie ook arnoud zijn redenering over waarom sentiment analysis niet per se nuttig is bij user profiles, enkel bij restaurant profiles)



sentiment analysis enkel op restaurant profile en NIET op user profile:
- restaurant heeft steak -> user zegt goed of niet goed => sentiment is belangrijk
- user zegt steak is niet legger bij restaurant X => user wil graag steak (dus wel recommenden), het was gewoon niet lekker bij dat restaurant


TODO is our recommender vulnerable to fake reviews (ja zeker, maar out of scope. Wel goed om te vermelden)