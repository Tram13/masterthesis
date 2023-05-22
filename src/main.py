import gc
import logging
import random
from pathlib import Path

from torch.optim import Adagrad
from tqdm import tqdm

from NLP.main_user_profiles import main_user_profile_topic
from NLP.utils.evaluate_model import evaluate_model
from NLP.utils.sentence_splitter import SentenceSplitter
from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from predictor.implementations.multilayer_perceptron import MultiLayerPerceptronPredictor
from predictor.implementations.multilayer_perceptron6 import MultiLayerPerceptron6Predictor
from predictor.neural_network_trainer import NeuralNetworkTrainer
from tools.restaurant_profiles_manager import RestaurantProfilesManager
from tools.user_profiles_manager import UserProfilesManager


# Code om een model te trainen, met de ingestelde parameters
# Optioneel kan een model en een optimizer meegegeven worden.
def main_train_models_with_same_data_splits(train_data, test_data, up_params, rp_params, EPOCHS, SUB_EPOCHS, models: list[MultiLayerPerceptronPredictor] = None, optimizers=None):
    with tqdm(total=EPOCHS, desc="Epochs", leave=True) as p_bar:
        train_test_data = DataPreparer.parse_data_train_test(train_data, test_data, (up_params, rp_params), cache_index_if_available=0)

        logging.info("Transforming data to DataLoaders")
        nn_trainer = NeuralNetworkTrainer(up_params, rp_params, *train_test_data)
        gc.collect()

        # If no models are provided, use the default implementation
        if models is None:
            lr = 0.0002
            logging.info("Using default Multi-Layer Perceptron model")
            models = [MultiLayerPerceptron6Predictor(nn_trainer.train_loader.dataset.x_train.shape[1])]  # Invullen met het juiste standaardmodel
            optimizers = [Adagrad(models[0].parameters(), lr)]  # Invullen met het juiste standaardmodel
            models[0].lr = lr
            models[0].optimizer = "ADAGRAD"

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


# Deze code zal verschillende combinaties van user profiles en restaurant profiles testen
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


# Code om guided topics: scores uit te rekenen
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


# Code om het offline BERTopic model te genereren
def main_user_profile_offline_bert():
    print("hello world")
    (_, reviews, _), _ = DataReader().read_data(no_train_test=True)

    logging.info('Finished reading in data, starting NLP...')

    logging.info(f'review size: {reviews.shape}')

    main_user_profile_topic(reviews,
                            amount_of_batches=20,
                            profile_name=f"offline_bert.parquet",
                            use_cache=False,
                            model_name="offline_bertopic_100000.bert",
                            use_sentiment_in_scores=False,
                            only_create_scores=True
                            )


# Clustering van een specifiek BERTopic model valideren
def main_evaluate_model(model_name):
    print("hello world")
    logging.basicConfig(level=logging.INFO)

    sentences = SentenceSplitter()._load_splitted_reviews_from_cache()

    logging.info('Finished reading in data, starting evaluation...')
    # logging.info("0.1% of the data")
    # evaluate_model(sentences, model_name, 1, True)
    # logging.info("0.5% of the data")
    # evaluate_model(sentences, model_name, 5, True)
    logging.info("1% of the data")
    evaluate_model(sentences, model_name, 1, False)
    # logging.info("2% of the data")
    # evaluate_model(sentences, model_name, 2, False)


def main():
    # Note: force manual garbage collection is used to save on memory after heavy RAM and I/O instructions
    EPOCHS = 20
    SUB_EPOCHS = 20  # Aantal keer dat met dezelfde NLP-profielen getraind wordt. Zie hoofdstuk 4: Dataflow

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

    # Dit traint een neuraal netwerk met de beste combinaties van inputfeatures en parameters voor het netwerk zelf.
    # De huidige instellingen geven het beste resultaat, met een MSE van 0.0771
    # Zie hoofdstuk 5 voor de analyses
    main_train_models_with_same_data_splits(train_data, test_data, up_params, rp_params, EPOCHS, SUB_EPOCHS)


if __name__ == '__main__':
    main()
