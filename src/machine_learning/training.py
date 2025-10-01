import sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from src.utils.protein_pair import ProteinPair
from src.machine_learning.evaluation import evaluate_classifier



def train_interaction_classifier(protein_pairs: list[ProteinPair], params):
    """Trains and evaluates a classifier for protein-protein interaction prediction.

    :param protein_pairs: List of ProteinPair objects with calculated features and labels.
    :param params: Dictionary containing parameters for training and evaluation.
    """

    X_train, X_test, y_train, y_test = prepare_train_test_data(protein_pairs)

    if params['classifier_type'] == 'random_forest':
        clf = random_forest_classifier(params)
    elif params['classifier_type'] == 'hist_gradient_boosting':
        clf = hist_gradient_boosting_classifier(params)

    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_

    evaluate_classifier(best_model, X_test, y_test, params['export_directory'], params['feature_list'])

    joblib.dump(best_model, f"{params['export_directory']}/best_model.pkl")
    print(f'Model saved to {params["export_directory"]}')

def prepare_train_test_data(protein_pairs: list[ProteinPair], split_ratio: float = 0.2):
    """Prepares the training and test datasets from a list of ProteinPair objects.

    :param protein_pairs: List of ProteinPair objects.
    :param split_ratio: Proportion of the dataset to include in the test split.
    :return: Tuple containing the training and test datasets.
    """
    features, labels = [], []
    for pair in protein_pairs:
        features.append(pair.features)
        labels.append(pair.label)

    return sklearn.model_selection.train_test_split(features, labels, test_size=split_ratio, random_state=2024)

def random_forest_classifier(params):
    """Creates a Random Forest classifier with the specified parameters.

    :param params: Dictionary containing parameters for the classifier.
    :return: Random Forest classifier.
    """
    param_grid = params['training_parameters']

    rf = RandomForestClassifier(random_state=1)

    # GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    return grid_search

def hist_gradient_boosting_classifier(params):
    """Creates a HistGradientBoosting classifier with the specified parameters.

    :param params: Dictionary containing parameters for the classifier.
    :return: HistGradientBoosting classifier.
    """
    param_grid = params['training_parameters']

    hgb = HistGradientBoostingClassifier(random_state=1)

    # GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=hgb,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    return grid_search



