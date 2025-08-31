import sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from src.utils.proteins import ProteinPair
from src.machine_learning.evaluation import evaluate_classifier
from src.machine_learning.evaluation import plot_baseline
def train_interaction_classifier(protein_pairs: list[ProteinPair], params):

    X_train, X_test, y_train, y_test = prepare_train_test_data(protein_pairs, export_directory=params['export_directory'], feature_list=params['feature_list'])

    clf = random_forest_classifier()
    #clf = hist_gradient_boosting_classifier()

    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_

    evaluate_classifier(best_model, X_test, y_test, params['export_directory'], params['feature_list'])

    joblib.dump(best_model, f"{params['export_directory']}/best_model.pkl")
    print(f'Model saved to {params["export_directory"]}')

def prepare_train_test_data(protein_pairs: list[ProteinPair], split_ratio: float = 0.2, feature_list=None, export_directory: str = None):
    """Prepares the training and test datasets from a list of ProteinPair objects.

    :param protein_pairs: List of ProteinPair objects.
    :param split_ratio: Proportion of the dataset to include in the test split.
    :return: Tuple containing the training and test datasets.
    """
    features, labels = [], []
    for pair in protein_pairs:
        features.append(pair.features)
        labels.append(pair.label)

    #plot_baseline(features, labels, feature_list, export_directory, dropna=True, filename="feature_baselines.png")

    return sklearn.model_selection.train_test_split(features, labels, test_size=split_ratio, random_state=240)

def random_forest_classifier():
    """Creates a Random Forest classifier with the specified parameters.

    :return: Random Forest classifier.
    """
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2', 0.5],
        'min_samples_split': [2, 5, 10]
    }
    # Todo: import param_grid from params.txt

    rf = RandomForestClassifier(random_state=1)

    # GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    return grid_search

def hist_gradient_boosting_classifier():
    """Creates a HistGradientBoosting classifier with the specified parameters.

    :return: HistGradientBoosting classifier.
    """
    param_grid = {
        'max_iter': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'learning_rate': [0.01, 0.1, 0.2],
        "min_samples_leaf": [10, 20, 50]
    }
    # Todo: import param_grid from params.txt

    hgb = HistGradientBoostingClassifier(random_state=42)

    # GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=hgb,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    return grid_search



