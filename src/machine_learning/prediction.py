import sklearn
import joblib
import pandas as pd
from src.utils.proteins import ProteinPair

def make_predictions(protein_pairs: list[ProteinPair], params):
    """Make predictions on a list of ProteinPair objects using a trained model.

    :param protein_pairs: List of ProteinPair objects with calculated features.
    :param params: Dictionary containing parameters, including the model filepath.
    :return: List of tuples containing (prefix, predicted_label, predicted_probability).
    """

    # Extract model filepath from parameters
    model_filepath = params['model_import_filepath']

    # Load the trained model
    clf = joblib.load(model_filepath)

    # Prepare features for prediction
    features = [pair.features for pair in protein_pairs]
    prefixes = [pair.prefix for pair in protein_pairs]

    # Make predictions
    predicted_labels = clf.predict(features)
    predicted_probabilities = clf.predict_proba(features)[:, 1]  # Probability of the positive class

    # Save results to csv using pandas
    with open(f"{params['export_directory']}/predictions.csv", 'w') as f:
        df = pd.DataFrame({
            'prefix': prefixes,
            'predicted_label': predicted_labels,
            'predicted_probability': predicted_probabilities
        })
        df.to_csv(f, index=False)

    print(f'Predictions saved to {params["export_directory"]}/predictions.csv')

    # Combine results into a list of tuples
    results = list(zip(prefixes, predicted_labels, predicted_probabilities))
    # Print results
    for prefix, label, prob in results:
        print(f'Prefix: {prefix}, Predicted Label: {label}, Predicted Probability: {prob:.4f}')