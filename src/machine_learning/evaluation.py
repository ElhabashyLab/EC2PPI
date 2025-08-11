import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, auc, roc_curve

def get_feature_names(feature_list):
    """Returns the feature names based on the provided feature list.

    :param feature_list: List of features to include.
    :return: List of feature names.
    """
    feature_names = []

    if 'n_eff' in feature_list:
        feature_names.append('n_eff_protein1')
        feature_names.append('n_eff_protein2')
    if 'n_eff_l' in feature_list:
        feature_names.append('n_eff_l_protein1')
        feature_names.append('n_eff_l_protein2')
    if 'sequence_length' in feature_list:
        feature_names.append('sequence_length_protein1')
        feature_names.append('sequence_length_protein2')
    if 'bit_score' in feature_list:
        feature_names.append('bit_score_protein1')
        feature_names.append('bit_score_protein2')
    if 'pairwise_identity' in feature_list:
        feature_names.append('pairwise_identity')
    if 'cn_mean' in feature_list:
        feature_names.append('cn_mean')
    if 'cn_std' in feature_list:
        feature_names.append('cn_std')
    if 'cn_median' in feature_list:
        feature_names.append('cn_median')
    if 'cn_iqr' in feature_list:
        feature_names.append('cn_iqr')
    if 'cn_max' in feature_list:
        feature_names.append('cn_max')
    if 'cn_skewness' in feature_list:
        feature_names.append('cn_skewness')
    if 'cn_kurtosis' in feature_list:
        feature_names.append('cn_kurtosis')

    return feature_names

def plot_roc_curve(fpr, tpr, roc_auc, export_directory):
    """Plots the ROC curve and saves it to the specified directory.

    :param fpr: False positive rates.
    :param tpr: True positive rates.
    :param roc_auc: Area under the ROC curve.
    :param export_directory: Directory to save the ROC curve plot.
    """
    if os.path.exists(f"{export_directory}/roc_curve.png"):
        os.remove(f"{export_directory}/roc_curve.png")

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
    plt.savefig(f"{export_directory}/roc_curve.png", dpi=300)
    plt.close()

    print(f"ROC curve saved as {export_directory}/roc_curve.png")

def plot_feature_importance(clf, feature_list, export_directory):
    """Plots the feature importance and saves it to the specified directory.

    :param clf: Trained classifier.
    :param feature_list: List of features used in the classifier.
    :param export_directory: Directory to save the feature importance plot.
    """
    feature_names = get_feature_names(feature_list)
    feature_importance = clf.feature_importances_
    indices = np.argsort(feature_importance)[::-1]

    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_feature_importance = feature_importance[indices]

    if os.path.exists(f"{export_directory}/feature_importance.png"):
        os.remove(f"{export_directory}/feature_importance.png")

    colors = ["#08306B" if i < 5 else "#9ECAE1" for i in range(len(sorted_feature_importance))]
    plt.figure(figsize=(8, 6))

    bars = plt.bar(range(len(sorted_feature_importance)), sorted_feature_importance, align='center', color=colors)
    plt.xticks(range(len(sorted_feature_importance)), sorted_feature_names, rotation=45, ha='right')
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{export_directory}/feature_importance.png", dpi=300)
    plt.close()

    print(f"Feature importance plot saved as {export_directory}/feature_importance.png")

    for i in range(len(feature_importance)):
        print(f"Feature {feature_names[indices[i]]}: importance = {feature_importance[indices[i]]:.4f}")

def evaluate_classifier(clf, X_test, y_test, export_directory, feature_list):
    """Evaluates the classifier on the test dataset.

    :param clf: Trained classifier.
    :param X_test: Test features.
    :param y_test: Test labels.
    :param export_directory: Directory to save evaluation results.
    :param feature_list: List of features used in the classifier.
    """

    y_prob = clf.predict_proba(X_test)[:, 1]

    print(f'Prob: {y_prob[0:10]} - True: {y_test[0:10]}')


    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, export_directory)

    # Print feature importance
    plot_feature_importance(clf, feature_list, export_directory)
