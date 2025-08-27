import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import precision_recall_curve

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
    if 'iptm' in feature_list:
        feature_names.append('iptm')
    if 'fraction_disordered' in feature_list:
        feature_names.append('fraction_disordered')
    if 'plddt_mean' in feature_list:
        feature_names.append('plddt_mean')
    if 'contact_probs_mean' in feature_list:
        feature_names.append('contact_probs_mean')
    if 'contact_probs_std' in feature_list:
        feature_names.append('contact_probs_std')
    if 'contact_probs_median' in feature_list:
        feature_names.append('contact_probs_median')
    if 'contact_probs_iqr' in feature_list:
        feature_names.append('contact_probs_iqr')
    if 'contact_probs_max' in feature_list:
        feature_names.append('contact_probs_max')
    if 'contact_probs_skewness' in feature_list:
        feature_names.append('contact_probs_skewness')
    if 'contact_probs_kurtosis' in feature_list:
        feature_names.append('contact_probs_kurtosis')
    if 'pae_mean' in feature_list:
        feature_names.append('pae_mean')
    if 'pae_std' in feature_list:
        feature_names.append('pae_std')
    if 'pae_median' in feature_list:
        feature_names.append('pae_median')
    if 'pae_iqr' in feature_list:
        feature_names.append('pae_iqr')
    if 'pae_max' in feature_list:
        feature_names.append('pae_max')
    if 'pae_skewness' in feature_list:
        feature_names.append('pae_skewness')
    if 'pae_kurtosis' in feature_list:
        feature_names.append('pae_kurtosis')

    return feature_names

def plot_baseline(
    X_test,
    y_test,
    feature_list=None,
    export_directory=".",
    feature_direction=None,   # e.g. {"PAE": "lower_is_better", "ipTM": "higher_is_better"}
    dropna=True,
    filename="baseline_roc.png"
):
    """
    Plot layered ROC curves using each feature in X_test as a score vs y_test.

    Parameters
    ----------
    X_test : pandas.DataFrame or np.ndarray (n_samples, n_features)
    y_test : array-like of shape (n_samples,)
    feature_list : list[str] | None
        If X_test is ndarray, provide names here; if DataFrame, uses df.columns by default.
    export_directory : str
    feature_direction : dict[str, "higher_is_better" | "lower_is_better"] | None
    dropna : bool
        If True, drop rows with NaN for each feature. If False, NaNs are filled with 0.
    filename : str
    """
    # Coerce to DataFrame for consistent column-wise iteration
    if isinstance(X_test, pd.DataFrame):
        df = X_test.copy()
        if feature_list is None:
            feature_list = list(df.columns)
        else:
            df = df[feature_list]  # subset to requested features
    else:
        X_test = np.asarray(X_test)
        if feature_list is None:
            feature_list = [f"f{i}" for i in range(X_test.shape[1])]
        if X_test.shape[1] != len(feature_list):
            raise ValueError("feature_list length must match X_test.shape[1]")
        df = pd.DataFrame(X_test, columns=feature_list)

    y = np.asarray(y_test).astype(int)
    if feature_direction is None:
        feature_direction = {}

    os.makedirs(export_directory, exist_ok=True)
    save_path = os.path.join(export_directory, filename)

    plt.figure(figsize=(12, 10))
    # random-chance diagonal
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="Random chance")

    aucs = {}

    for feat in feature_list:
        s = df[feat].astype(float)

        # NaN handling per feature
        if dropna:
            mask = ~s.isna()
            s_ = s[mask].values
            y_ = y[mask]
        else:
            s_ = s.fillna(0.0).values
            y_ = y

        # Need both classes to compute ROC
        if len(np.unique(y_)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_, s_)
        roc_auc = auc(fpr, tpr)

        # If AUC < 0.5, flip scores
        if roc_auc < 0.5:
            s_ = -s_
            fpr, tpr, _ = roc_curve(y_, s_)
            roc_auc = auc(fpr, tpr)

        aucs[feat] = roc_auc
        plt.plot(fpr, tpr, label=f"{feat} (AUC={roc_auc:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC – single-feature baselines")
    # Legend outside, save with tight bbox so it isn't cut off
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Baseline ROC saved to {save_path}")
    return aucs

def plot_roc_curve(y_test, y_prob, export_directory):
    """Plots the ROC curve and saves it to the specified directory.

    :param y_test: True labels.
    :param y_prob: Predicted probabilities.
    :param export_directory: Directory to save the ROC curve plot.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    if os.path.exists(f"{export_directory}/roc_curve.png"):
        os.remove(f"{export_directory}/roc_curve.png")

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
    plt.savefig(f"{export_directory}/roc_curve.png", dpi=300)
    plt.close()

    print(f"ROC curve saved as {export_directory}/roc_curve.png")

    return fpr, tpr, roc_auc

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
    plt.figure(figsize=(8, 5))

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

    return sorted_feature_names, sorted_feature_importance

def plot_precision_recall_curve(y_true, y_scores, export_directory):
    """Plots the precision-recall curve and saves it to the specified directory.

    :param y_true: True labels.
    :param y_scores: Predicted scores.
    :param export_directory: Directory to save the precision-recall curve plot.
    """

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    if os.path.exists(f"{export_directory}/precision_recall_curve.png"):
        os.remove(f"{export_directory}/precision_recall_curve.png")

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(f"{export_directory}/precision_recall_curve.png", dpi=300)
    plt.close()

    print(f"Precision-recall curve saved as {export_directory}/precision_recall_curve.png")

    return precision, recall

def save_evaluation_results(export_directory, fpr, tpr, roc_auc, precision, recall, feature_names, feature_importance):
    """Saves the evaluation results to a JSON file.

    :param export_directory: Directory to save the evaluation results.
    :param fpr: False positive rates.
    :param tpr: True positive rates.
    :param roc_auc: Area under the ROC curve.
    :param precision: Precision values.
    :param recall: Recall values.
    :param feature_names: Names of the features used in the classifier.
    :param feature_importance: Importance of the features used in the classifier.
    """

    evaluation_results = {
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": roc_auc
        },
        "precision_recall_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist()
        },
        "feature_importance": {
            "feature_names": feature_names,
            "importance": feature_importance.tolist()
        }
    }

    with open(f"{export_directory}/evaluation_results.json", 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"Evaluation results saved as {export_directory}/evaluation_results.json")

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


    # Generate and print ROC curve
    fpr, tpr, roc_auc = plot_roc_curve(y_test, y_prob, export_directory)

    # Print precision-recall curve
    precision, recall = plot_precision_recall_curve(y_test, y_prob, export_directory)

    # Print feature importance
    sorted_feature_list, sorted_feature_importance = plot_feature_importance(clf, feature_list, export_directory)

    # Save evaluation results
    save_evaluation_results(export_directory, fpr, tpr, roc_auc, precision, recall, sorted_feature_list, sorted_feature_importance)


