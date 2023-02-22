from src.data.data_reader import DataReader

print("hello world")
businesses, checkins, reviews, tips, users = DataReader().read_data()
print(businesses)
