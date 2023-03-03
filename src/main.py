from src.NLP.main_online_BERTopic import create_model_online_BERTopic, create_scores_from_online_model
from src.NLP.sentence_splitter import SentenceSplitter
from src.data.data_preparer import DataPreparer
from src.data.data_reader import DataReader


def main():
    print("hello world")
    _, reviews, _ = DataReader().read_data()
    # print(DataPreparer.get_train_test_validate(businesses, reviews, tips))

    print('Finished reading in data, starting NLP...')
    # create a fitted online model with the data
    create_model_online_BERTopic(reviews['text'])

    # gather the scores based of the current model
    # scores = create_scores_from_online_model(reviews['text'])


if __name__ == '__main__':
    main()
