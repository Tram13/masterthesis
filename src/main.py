import gc
import logging
import random
from pathlib import Path

from torch import optim
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

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


def single_model_multiprocessed(args):
    nn_trainer, models, optimizers, index, SUB_EPOCHS, EPOCHS = args
    model, optimizer = nn_trainer.train(models[index], optimizers[index], sub_epochs=SUB_EPOCHS, save_to_disk=(EPOCHS == 1), verbose=False)
    models[index] = model
    optimizers[index] = optimizer
    logging.info(f"Current loss history: {[f'{val:.3}' for val in model.loss_history[-5:]]}")


def main_single_model(train_data, test_data, up_params, rp_params, EPOCHS, SUB_EPOCHS, LR, multiprocess: bool = False):
    with tqdm(total=EPOCHS, desc="Epochs", leave=False) as p_bar:
        train_test_data = DataPreparer.parse_data_train_test(train_data, test_data, (up_params, rp_params), cache_index_if_available=0)

        logging.info("Transforming data to DataLoaders")
        nn_trainer = NeuralNetworkTrainer(up_params, rp_params, *train_test_data)
        gc.collect()

        logging.info("Creating Multi-Layer Perceptron models")
        models = [
            MultiLayerPerceptron1Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron2Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron3Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron4Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron5Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron6Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron7Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron8Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1])
        ]
        optimizers = [  # TODO: onderzoek de optimizer!
            optim.SGD(models[0].parameters(), lr=LR),
            optim.SGD(models[1].parameters(), lr=LR),
            optim.SGD(models[2].parameters(), lr=LR),
            optim.SGD(models[3].parameters(), lr=LR),
            optim.SGD(models[4].parameters(), lr=LR),
            optim.SGD(models[5].parameters(), lr=LR),
            optim.SGD(models[6].parameters(), lr=LR),
            optim.SGD(models[7].parameters(), lr=LR)
        ]

        # Setting default parameter correctly
        for model in models:
            model.optimizer = "SGD"
            model.lr = LR

        models.extend((
            MultiLayerPerceptron5Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron6Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron5Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron6Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron5Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1]),
            MultiLayerPerceptron6Predictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1])
        ))

        optimizers.extend((
            optim.Adagrad(models[8].parameters(), lr=LR),
            optim.Adagrad(models[9].parameters(), lr=LR),
            optim.Adagrad(models[10].parameters(), lr=0.002),
            optim.Adagrad(models[11].parameters(), lr=0.002),
            optim.Adagrad(models[12].parameters(), lr=0.02),
            optim.Adagrad(models[13].parameters(), lr=0.02),
        ))

        models[8].lr = LR
        models[9].lr = LR
        models[10].lr = 0.002
        models[11].lr = 0.002
        models[12].lr = 0.02
        models[13].lr = 0.02

        logging.info("Starting training")
        if multiprocess:
            args = [(nn_trainer, models, optimizers, index, SUB_EPOCHS, EPOCHS) for index in range(len(models))]
            process_map(single_model_multiprocessed, args, max_workers=5)
        else:
            for index, (model, optimizer) in tqdm(enumerate(zip(models, optimizers)), desc="Training models", leave=False):
                if index == 9:
                    model, optimizer = nn_trainer.train(model, optimizer, sub_epochs=SUB_EPOCHS, save_to_disk=(EPOCHS == 1), verbose=True)
                    models[index] = model
                    optimizers[index] = optimizer
                    logging.info(f"Current loss history: {[f'{val:.3}' for val in model.loss_history[-5:]]}")
        p_bar.update()
        logging.info(f"Validation check: {models[11].current_epoch} epochs in models[2]")

        for epoch in range(2, EPOCHS + 1):
            gc.collect()  # Clean up last run
            # Creates new user profiles based on subset of train or test data
            train_test_data = DataPreparer.parse_data_train_test(train_data, test_data, (up_params, rp_params), cache_index_if_available=epoch-1)

            logging.info("Transforming data to DataLoaders")
            nn_trainer = NeuralNetworkTrainer(up_params, rp_params, *train_test_data)
            gc.collect()

            logging.info("Starting training")
            if multiprocess:
                args = [(nn_trainer, models, optimizers, index, SUB_EPOCHS, EPOCHS) for index in range(len(models))]
                process_map(single_model_multiprocessed, args, max_workers=4)
            else:
                for index, (model, optimizer) in tqdm(enumerate(zip(models, optimizers)), desc="Training models", leave=False):
                    if index == 9:
                        model, optimizer = nn_trainer.train(model, optimizer, sub_epochs=SUB_EPOCHS, save_to_disk=(EPOCHS == epoch), verbose=True)
                        models[index] = model
                        optimizers[index] = optimizer
                        logging.info(f"Current loss history: {[f'{val:.3}' for val in model.loss_history[-5:]]}")
            logging.info(f"Validation check: {models[11].current_epoch} epochs in models[2]")
            p_bar.update()

        # Save statistics
    for model in models:
        model.plot_loss_progress(save_location=Path(f"{str(model.get_default_save_location())[:-3]}.png"))
    gc.collect()


def main_profiles_grid_search_for_nn(EPOCHS: int, SUB_EPOCHS: int, LR: int):
    logging.info("Reading Yelp Dataset")
    train_data, test_data = DataReader().read_data()
    gc.collect()
    for user_index, up_params in enumerate(tqdm(UserProfilesManager(), desc="User Profiles")):
        for restaurant_index, rp_params in enumerate(tqdm(RestaurantProfilesManager(), desc="Restaurant Profiles")):
            # Checking for duplicate work
            with open("done_combinations.txt", mode='r', encoding='utf-8') as done_file:
                combos_done = [line.rstrip() for line in done_file.readlines()]
            # If not found yet
            if f"{user_index}_{restaurant_index}" not in combos_done:
                if random.random() <= 0.3:  # Chance to skip this configuration
                    logging.warning(f"Randomly skipped model {(user_index, restaurant_index)}")
                    continue
                logging.info(f"Running model {(user_index, restaurant_index)}")
                main_single_model(train_data, test_data, up_params, rp_params, EPOCHS, SUB_EPOCHS, LR)
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
    logging.basicConfig(level=logging.INFO)

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


if __name__ == '__main__':
    # Note: force manual garbage collection is used to save on memory after heavy RAM and I/O instructions
    EPOCHS = 10
    SUB_EPOCHS = 40
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
    main_single_model(train_data, test_data, up_params, rp_params, EPOCHS, SUB_EPOCHS, LR)

    # logging.info("***************** splits maken *****************")
    # DataPreparer.make_nn_caches(train_data, test_data, up_params, rp_params, n=50)
    # logging.info("***************** random forest trainen *****************")
    # main_random_forest(train_data, test_data, up_params, rp_params)
