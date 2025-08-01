import sys
from src.data_processing.read_params import read_params
import src.data_processing.read_dataset as read_dataset
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

    # Read the dataset based on the parameters
    if params['training_run']:
        training_protein_pairs = read_dataset.read_training_dataset(params)
        print(f'Read {len(training_protein_pairs)} protein pairs for training.')
        #for pair in training_protein_pairs:
            #print(f'Protein Pair: {pair.protein1._uid} - {pair.protein2._uid}')
        # TODO

        # Calculate features for the training dataset
        # training_data = calculate_features(training_protein_pairs, params)

        # Initialize the predictor and train the model
        # predictor = Predictor(params)
        # predictor.train(training_data)
        # predictor.save_model(params['model_export_filepath'])

    #if params['prediction_run']:

        # prediction_protein_pairs = read_dataset.read_applied_dataset(params)

        # Calculate features for the prediction dataset
        # prediction_data = calculate_features(prediction_protein_pairs, params)
        # predictor = Predictor(params)
        # predictor.load_model(params['model_import_filepath'])
        # predictions = predictor.predict(prediction_data)
        # predictor.export_predictions(predictions, params['prediction_export_filepath'])


if __name__ == "__main__":
    main()


