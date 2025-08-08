import sklearn
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, auc, roc_curve
from src.utils.protein_pair import ProteinPair
def train_interaction_classifier(protein_pairs: list[ProteinPair], params):

    X_train, X_test, y_train, y_test = prepare_train_test_data(protein_pairs)

    clf = random_forest_classifier()
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    evaluate_classifier(best_model, X_test, y_test)

    joblib.dump(best_model, params['model_export_filepath'])

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

    return sklearn.model_selection.train_test_split(features, labels, test_size=split_ratio, random_state=42)

def random_forest_classifier():
    """Creates a Random Forest classifier with the specified parameters.

    :return: Random Forest classifier.
    """
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)

    # GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    return grid_search

def evaluate_classifier(clf, X_test, y_test):
    """Evaluates the classifier on the test dataset.

    :param clf: Trained classifier.
    :param X_test: Test features.
    :param y_test: Test labels.
    :return: Dictionary containing evaluation metrics.
    """

    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f'Prob: {y_prob[0:10]} - True: {y_test[0:10]}')

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    feature_importance = clf.feature_importances_
    indices = np.argsort(feature_importance)[::-1]

    for i in range(len(feature_importance)):
        print(f"Feature {indices[i]}: importance = {feature_importance[indices[i]]:.4f}")

    print(classification_report(y_test, clf.predict(X_test)))



