import gc
import logging
from pathlib import Path

from torch import optim

from NLP.profiles_creator import ProfileCreator
from data.data_preparer import DataPreparer
from data.data_reader import DataReader
from predictor.multilayer_perceptron import MultiLayerPerceptronPredictor
from predictor.neural_network_trainer import NeuralNetworkTrainer


def get_data_train_test():
    logging.info("Reading Yelp Dataset")
    (b_train, r_train, u_train), (b_test, r_test, u_test) = DataReader().read_data()
    gc.collect()
    logging.info("Parsing train set")
    training_input, training_output, users_param, businesses_param = transform_data(b_train, r_train, u_train)
    logging.info("Parsing test set")
    test_input, test_output, _, _ = transform_data(b_test, r_test, u_test)
    gc.collect()
    return (training_input, test_input, training_output, test_output), (users_param, businesses_param)


def transform_data(businesses, reviews, users):
    logging.info("Splitting in generation and prediction sets")
    reviews_generation, reviews_prediction = DataPreparer.get_profiles_split(reviews, profile_dataframe_size=0.7)

    logging.info("Creating User Profile")
    profile_creator = ProfileCreator(
        model_name="online_model_400top_97.bert",
        use_sentiment_in_scores=False,
        approx_mode=False,
        approx_normalization=True,
        approx_amount_top_n=5,
        filter_useful_topics=False
    )
    user_profiles_nlp = profile_creator.get_user_profile(reviews_generation)
    user_profiles_parameters = profile_creator.get_parameters_string()

    logging.info("Creating Restaurant Profile")
    profile_creator = ProfileCreator(
        model_name="online_model_50top_85.bert",
        use_sentiment_in_scores=True,
        approx_mode=True,
        approx_normalization=True,
        approx_amount_top_n=5,
        filter_useful_topics=False
    )
    business_profiles_nlp = profile_creator.get_restaurant_profile(reviews_generation)
    business_profiles_parameters = profile_creator.get_parameters_string()

    logging.info("Transforming to ML input")
    input_ml, output_ml = DataPreparer.get_df_for_ml(businesses, reviews_prediction, users, user_profiles_nlp, business_profiles_nlp)
    gc.collect()
    return input_ml, output_ml, user_profiles_parameters, business_profiles_parameters


def main_single_model():
    # Parameters
    EPOCHS = 1
    LR = 0.0002

    logging.info("Initialising")
    train_test_data, (users_params, businesses_params) = get_data_train_test()
    nn_trainer = NeuralNetworkTrainer(users_params, businesses_params, *train_test_data)
    logging.info("Cleaning up memory on CPU")
    gc.collect()

    logging.info("Creating Multi-Layer Perceptron model")
    model = MultiLayerPerceptronPredictor(input_size=nn_trainer.train_loader.dataset.x_train.shape[1], output_size=1)
    optimizer = optim.Adagrad(model.parameters(), lr=LR)

    # Training on first part of dataset
    logging.info("Starting training")
    model, optimizer = nn_trainer.train(model, optimizer, epochs=EPOCHS, save_to_disk=True, verbose=True)

    # Save statistics
    model.plot_loss_progress(save_location=Path(f"{model.get_default_save_location()[:-2]}.png"))
    return 0


if __name__ == '__main__':
    # Note: force manual garbage collection is used to save on memory after heavy RAM and I/O instructions
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%H:%M:%S',
        format='%(asctime)s %(levelname)-8s %(message)s',
    )
    main_single_model()
