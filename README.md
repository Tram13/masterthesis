# Design van een machine learning-algoritme voor aanbevelingen van restaurants op basis van gelabelde en tekstuele data
Auteur: Arnoud De Jonge, Arno Vermote\
Promotor: prof. dr. ir. Toon De Pessemier, prof. dr. ir. Luc Martens\
Begeleider: prof. dr. ir. Toon De Pessemier

Masterproef ingediend tot het behalen van de academische graad van Master of Science in de Informatica\
Academiejaar 2022-2023

Contact: [arvermot.vermote@ugent.be](mailto:arvermot.vermote@ugent.be)

## Inleiding
Deze repository bevat alle code en data waarmee wij onze experimenten uitvoerden. De code staat in `src`. De onverwerkte data staat in `data`.\
Alle overige informatie, zoals de volledige thesis in PDF en LaTeX-vorm, Excel-bestanden voor grafieken en PowerPoint van de tusesntijdse presentatie staan in `docs`.

In `src` staat een README.md die de verdere uitleg voor de code uitlegt.

## Setup en uitvoering
1. Installeer python3.10
2. Installeer de nodige python packages aan de hand van `requirements.txt`. Support voor NVIDIA CUDA is sterk aangeraden als nieuwe neurale netwerken of NLP-profielen moeten opgesteld worden.
3. Pas de configwaarden aan in `config.ini` (optioneel)
4. Voer `python main.py` uit vanuit de `src`-directory van dit project.
