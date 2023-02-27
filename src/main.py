from src.data.data_reader import DataReader
import torch

def main():
    print("hello world")
    businesses, reviews, tips, users = DataReader().read_data()
    print(torch.cuda.is_available())


if __name__ == '__main__':
    main()
