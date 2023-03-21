import logging

from src.NLP.main_zero_shot_classification import main_calculate_zero_shot_classification
from src.data.data_reader import DataReader


def main():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    reviews = reviews.head(100)

    logging.info('Finished reading in data, starting NLP...')
    basic_classes = ["food", "service", "environment"]

    main_calculate_zero_shot_classification(reviews['text'], classes=basic_classes)


if __name__ == '__main__':
    main()
