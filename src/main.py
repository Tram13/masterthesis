import logging

from NLP.main_online_BERTopic import create_model_online_BERTopic
from NLP.main_user_profiles import main_user_profile_approximation
from NLP.utils.sentence_splitter import SentenceSplitter
from data.data_reader import DataReader


def main_bert_model_200topics():
    pass


def main():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    _, reviews, _ = DataReader().read_data()
    # reviews = reviews.head(1000)

    logging.info('Finished reading in data, starting NLP...')
    main_user_profile_approximation(reviews)


if __name__ == '__main__':
    main()
