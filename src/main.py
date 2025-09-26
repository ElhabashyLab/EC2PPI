import sys
import time
from src.data_processing.read_params import read_params
import src.data_processing.read_dataset as read_dataset
from src.machine_learning.training import train_interaction_classifier
from src.data_processing.feature_calculation import calculate_all_features
from src.machine_learning.prediction import make_predictions
def main():
    """Main function,wraps the main logic of the program.
    """

    # Check if the params file is given as argument
    args = sys.argv
    if len(args) != 2:
        raise ValueError(f'WARNING: Expected 1 argument but received {len(args) - 1}. Please provide the absolute or relative filepath to \'params_file.txt\'.')
    params_file_path = args[1]

    # Read the parameters from the params file
    params = read_params(params_file_path)

    start_time = time.time()

    if params['training_run']:

        # Read the training dataset
        training_protein_pairs = read_dataset.read_training_dataset(params)

        read_in_time = time.time()
        print(f"Time taken to read dataset: {read_in_time - start_time:.2f} seconds")
        print(f"Total time so far: {read_in_time - start_time:.2f} seconds")

        # smaller debug sets
        #training_protein_pairs = training_protein_pairs[:1]  # Use only the first protein pair for debugging
        #training_protein_pairs = training_protein_pairs[650:-650]

        # Calculate features for the training dataset
        calculate_all_features(training_protein_pairs, params)

        feature_calculation_time = time.time()
        print(f"Time taken to calculate features: {feature_calculation_time - read_in_time:.2f} seconds")
        print(f"Total time so far: {feature_calculation_time - start_time:.2f} seconds")

        # Train and evaluate the interaction classifier
        train_interaction_classifier(training_protein_pairs, params)

        training_evaluation_time = time.time()
        print(f"Time taken to train and evaluate model: {training_evaluation_time - feature_calculation_time:.2f} seconds")
        print(f"Total time taken: {training_evaluation_time - start_time:.2f} seconds")




    if params['prediction_run']:

        # Read the prediction dataset
        prediction_protein_pairs = read_dataset.read_applied_dataset(params)

        read_in_time = time.time()
        print(f"Time taken to read dataset: {read_in_time - start_time:.2f} seconds")
        print(f"Total time so far: {read_in_time - start_time:.2f} seconds")

        # smaller debug sets
        #prediction_protein_pairs = prediction_protein_pairs[:1]  # Use only the first protein pair for debugging
        #rediction_protein_pairs = prediction_protein_pairs[300:-300]

        # Calculate features for the prediction dataset
        calculate_all_features(prediction_protein_pairs, params)

        feature_calculation_time = time.time()
        print(f"Time taken to calculate features: {feature_calculation_time - read_in_time:.2f} seconds")
        print(f"Total time so far: {feature_calculation_time - start_time:.2f} seconds")

        # Predict interactions using the trained model
        make_predictions(prediction_protein_pairs, params)

        prediction_time = time.time()
        print(f"Time taken for prediction: {prediction_time - feature_calculation_time:.2f} seconds")
        print(f"Total time taken: {prediction_time - start_time:.2f} seconds")




if __name__ == "__main__":
    main()


