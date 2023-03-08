from src.data.data_preparer import DataPreparer
from src.data.data_reader import DataReader
import torch

def main():
    print("hello world")
    businesses, reviews, tips = DataReader().read_data()
    print(DataPreparer.get_train_test_validate(businesses, reviews, tips))


if __name__ == '__main__':
    main()
