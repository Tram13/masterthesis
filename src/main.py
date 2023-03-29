import logging

from NLP.main_online_BERTopic import create_model_online_BERTopic
from data.data_reader import DataReader


def main():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    # reviews = reviews.head(1000)

    logging.info('Finished reading in data, starting NLP...')

    create_model_online_BERTopic(reviews['text'])


if __name__ == '__main__':
    main()
