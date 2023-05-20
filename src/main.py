import gc
import logging
import random
from pathlib import Path

from torch import optim
from torch.optim import Adagrad
from tqdm import tqdm

from NLP.main_user_profiles import main_user_profile_topic
from NLP.utils.evaluate_model import evaluate_model
from NLP.utils.sentence_splitter import SentenceSplitter
from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from predictor.implementations.multilayer_perceptron1 import MultiLayerPerceptron1Predictor
from predictor.implementations.multilayer_perceptron2 import MultiLayerPerceptron2Predictor
from predictor.implementations.multilayer_perceptron3 import MultiLayerPerceptron3Predictor
from predictor.implementations.multilayer_perceptron4 import MultiLayerPerceptron4Predictor
from predictor.implementations.multilayer_perceptron5 import MultiLayerPerceptron5Predictor
from predictor.implementations.multilayer_perceptron6 import MultiLayerPerceptron6Predictor
from predictor.implementations.multilayer_perceptron7 import MultiLayerPerceptron7Predictor
from predictor.implementations.multilayer_perceptron8 import MultiLayerPerceptron8Predictor
from predictor.neural_network_trainer import NeuralNetworkTrainer
from tools.restaurant_profiles_manager import RestaurantProfilesManager
from tools.user_profiles_manager import UserProfilesManager


def main_train_models_with_same_data_splits(train_data, test_data, up_params, rp_params, EPOCHS, SUB_EPOCHS, models=None, optimizers=None):
    with tqdm(total=EPOCHS, desc="Epochs", leave=True) as p_bar:
        train_test_data = DataPreparer.parse_data_train_test(train_data, test_data, (up_params, rp_params), cache_index_if_available=0)

        logging.info("Transforming data to DataLoaders")
        nn_trainer = NeuralNetworkTrainer(up_params, rp_params, *train_test_data)
        gc.collect()

        # If no models are provided, use the default implementation
        if models is None:
            logging.info("Using default Multi-Layer Perceptron model")
            models = [MultiLayerPerceptron5Predictor(nn_trainer.train_loader.dataset.x_train.shape[1])]  # Invullen met het juiste standaardmodel
            optimizers = [Adagrad(models[0].parameters(), 0.0002)]  # Invullen met het juiste standaardmodel

        logging.info("Starting training")
        for index, (model, optimizer) in tqdm(enumerate(zip(models, optimizers)), desc="Training models", leave=False, total=len(models)):
            model, optimizer = nn_trainer.train(model, optimizer, sub_epochs=SUB_EPOCHS, save_to_disk=(EPOCHS == 1), verbose=True)
            models[index] = model
            optimizers[index] = optimizer
            logging.info(f"Current loss history: {[f'{val:.3}' for val in model.loss_history[-5:]]}")
        p_bar.update()

        for epoch in range(2, EPOCHS + 1):
            gc.collect()  # Clean up last run
            # Creates new user profiles based on subset of train or test data
            train_test_data = DataPreparer.parse_data_train_test(train_data, test_data, (up_params, rp_params), cache_index_if_available=epoch - 1)

            logging.info("Transforming data to DataLoaders")
            nn_trainer = NeuralNetworkTrainer(up_params, rp_params, *train_test_data)
            gc.collect()

            logging.info("Starting training")
            for index, (model, optimizer) in tqdm(enumerate(zip(models, optimizers)), desc="Training models", leave=False, total=len(models)):
                model, optimizer = nn_trainer.train(model, optimizer, sub_epochs=SUB_EPOCHS, save_to_disk=(EPOCHS == epoch), verbose=True)
                models[index] = model
                optimizers[index] = optimizer
                logging.info(f"Current loss history: {[f'{val:.3}' for val in model.loss_history[-5:]]}")
            p_bar.update()

        # Save statistics
    for model in models:
        model.plot_loss_progress(save_location=Path(f"{str(model.get_default_save_location())[:-3]}.png"))
    gc.collect()


def main_profiles_grid_search_for_nn(EPOCHS: int, SUB_EPOCHS: int):
    logging.info("Reading Yelp Dataset")
    train_data_f, test_data_f = DataReader().read_data()
    gc.collect()
    for user_index, up_param in enumerate(tqdm(UserProfilesManager(), desc="User Profiles")):
        for restaurant_index, rp_param in enumerate(tqdm(RestaurantProfilesManager(), desc="Restaurant Profiles")):
            # Checking for duplicate work
            with open("done_combinations.txt", mode='r', encoding='utf-8') as done_file:
                combos_done = [line.rstrip() for line in done_file.readlines()]
            # If not found yet
            if f"{user_index}_{restaurant_index}" not in combos_done:
                if random.random() <= 0.3:  # Chance to skip this configuration
                    logging.warning(f"Randomly skipped model {(user_index, restaurant_index)}")
                    continue
                logging.info(f"Running model {(user_index, restaurant_index)}")
                main_train_models_with_same_data_splits(train_data_f, test_data_f, up_param, rp_param, EPOCHS, SUB_EPOCHS)
                with open("done_combinations.txt", mode='a+', encoding='utf-8') as done_combinations:
                    done_combinations.write(f"{user_index}_{restaurant_index}\n")
                exit(1)  # To clear up memory
            else:  # Skip
                logging.warning(f"Skipped model {(user_index, restaurant_index)}")
    return 0


def main_guided_topics_score_creation():
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    (_, reviews, _), _ = DataReader().read_data(no_train_test=True)

    logging.info('Finished reading in data, starting NLP...')

    logging.info('User profile')
    main_user_profile_topic(reviews,
                            profile_name="GUIDED_USER_58.parquet",
                            model_name="BERTopic_guided_maxtop_58.bert",
                            use_sentiment_in_scores=False,
                            only_create_scores=True
                            )


def main_user_profile_offline_bert():
    print("hello world")
    (_, reviews, _), _ = DataReader().read_data(no_train_test=True)

    logging.info('Finished reading in data, starting NLP...')

    logging.info(f'review size: {reviews.shape}')

    main_user_profile_topic(reviews,
                            amount_of_batches=10,
                            profile_name=f"offline_bert.parquet",
                            use_cache=False,
                            model_name="offline_bertopic_100000.bert",
                            use_sentiment_in_scores=False,
                            only_create_scores=True
                            )


def main_evaluate_model(model_name):
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    sentences = SentenceSplitter()._load_splitted_reviews_from_cache()

    logging.info('Finished reading in data, starting evaluation...')
    logging.info("0.1% of the data")
    evaluate_model(sentences, model_name, 1, True)
    logging.info("0.5% of the data")
    evaluate_model(sentences, model_name, 5, True)
    logging.info("1% of the data")
    evaluate_model(sentences, model_name, 1, False)
    logging.info("2% of the data")
    evaluate_model(sentences, model_name, 2, False)


def main():
    # Note: force manual garbage collection is used to save on memory after heavy RAM and I/O instructions
    EPOCHS = 20
    SUB_EPOCHS = 20
    LR = 0.0002

    logging.basicConfig(
        level=logging.INFO,
        datefmt='%H:%M:%S',
        format='%(asctime)s %(levelname)-8s %(message)s',
    )

    logging.info("Reading Yelp Dataset")
    train_data, test_data = DataReader().read_data()
    up_params = UserProfilesManager().get_best()
    rp_params = RestaurantProfilesManager().get_best()
    gc.collect()

    logging.info("***************** alle netwerkconfiguraties proberen *****************")
    dummy_data = DataPreparer.parse_data_train_test(train_data, test_data, (up_params, rp_params), cache_index_if_available=0)
    dummy_trainer = NeuralNetworkTrainer(up_params, rp_params, *dummy_data)
    models = [
        MultiLayerPerceptron1Predictor(input_size=dummy_trainer.train_loader.dataset.x_train.shape[1]),
        MultiLayerPerceptron2Predictor(input_size=dummy_trainer.train_loader.dataset.x_train.shape[1]),
        MultiLayerPerceptron3Predictor(input_size=dummy_trainer.train_loader.dataset.x_train.shape[1]),
        MultiLayerPerceptron4Predictor(input_size=dummy_trainer.train_loader.dataset.x_train.shape[1]),
        MultiLayerPerceptron5Predictor(input_size=dummy_trainer.train_loader.dataset.x_train.shape[1]),
        MultiLayerPerceptron6Predictor(input_size=dummy_trainer.train_loader.dataset.x_train.shape[1]),
        MultiLayerPerceptron7Predictor(input_size=dummy_trainer.train_loader.dataset.x_train.shape[1]),
        MultiLayerPerceptron8Predictor(input_size=dummy_trainer.train_loader.dataset.x_train.shape[1])
    ]
    optimizers = [
        optim.Adagrad(models[0].parameters(), lr=LR),
        optim.Adagrad(models[1].parameters(), lr=LR),
        optim.Adagrad(models[2].parameters(), lr=LR),
        optim.Adagrad(models[3].parameters(), lr=LR),
        optim.Adagrad(models[4].parameters(), lr=LR),
        optim.Adagrad(models[5].parameters(), lr=LR),
        optim.Adagrad(models[6].parameters(), lr=LR),
        optim.Adagrad(models[7].parameters(), lr=LR)
    ]
    # Setting default parameter correctly
    for i_model in models:
        i_model.optimizer = "ADAGRAD"
        i_model.lr = LR

    main_train_models_with_same_data_splits(train_data, test_data, up_params, rp_params, EPOCHS, SUB_EPOCHS, models, optimizers)

    logging.info("***************** Coldstart *****************")
    # TODO: zorgen dat we een getraind model pakken, en dan evalueren met een subset van de data, id est de coldstart dataset.
    train_data, test_data = DataReader().read_data(at_least=5)
    # TODO: vergeet zeker niet te zorgen dat de caching uitstaat bij de profile split generation! de cache_index_if_available of whatever!!





    logging.info("***************** splits maken *****************")
    DataPreparer.make_nn_caches(train_data, test_data, up_params, rp_params, resume_from=33, n=50)
    # logging.info("***************** random forest trainen *****************")
    # main_random_forest(train_data, test_data, up_params, rp_params)


if __name__ == '__main__':
    main()
