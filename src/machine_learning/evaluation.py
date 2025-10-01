import matplotlib.pyplot as plt
import numpy as np
import os
import json
from sklearn.metrics import auc, roc_curve, average_precision_score
from sklearn.metrics import precision_recall_curve


def plot_roc_curve(y_test, y_prob, export_directory):
    """Plots the ROC curve and saves it to the specified directory.

    :param y_test: True labels.
    :param y_prob: Predicted probabilities.
    :param export_directory: Directory to save the ROC curve plot.
    """

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    if os.path.exists(f"{export_directory}/roc_curve.png"):
        os.remove(f"{export_directory}/roc_curve.png")

    fs = 15
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#3668B2")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate", fontsize=fs)
    plt.ylabel("True Positive Rate", fontsize=fs)
    plt.title("ROC Curve", fontsize=fs)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=fs-2)
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

    feature_importance = clf.feature_importances_
    indices = np.argsort(feature_importance)[::-1]

    sorted_feature_names = [feature_list[i] for i in indices]
    sorted_feature_importance = feature_importance[indices]

    if os.path.exists(f"{export_directory}/feature_importance.png"):
        os.remove(f"{export_directory}/feature_importance.png")

    colors = ["#08306B" if i < 5 else "#9ECAE1" for i in range(len(sorted_feature_importance))]
    plt.figure(figsize=(8, 5))

    plt.bar(range(len(sorted_feature_importance)), sorted_feature_importance, align='center', color=colors)
    plt.xticks(range(len(sorted_feature_importance)), sorted_feature_names, rotation=45, ha='right')
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{export_directory}/feature_importance.png", dpi=300)
    plt.close()

    print(f"Feature importance plot saved as {export_directory}/feature_importance.png")

    for i in range(len(feature_importance)):
        print(f"Feature {feature_list[indices[i]]}: importance = {feature_importance[indices[i]]:.4f}")

    return sorted_feature_names, sorted_feature_importance

def plot_precision_recall_curve(y_true, y_scores, export_directory):
    """Plots the precision-recall curve and saves it to the specified directory.

    :param y_true: True labels.
    :param y_scores: Predicted scores.
    :param export_directory: Directory to save the precision-recall curve plot.
    """

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1_idx = np.argmax(f1_scores)
    f1_threshold = thresholds[f1_idx] if f1_idx < len(thresholds) else 1.0

    if os.path.exists(f"{export_directory}/precision_recall_curve.png"):
        os.remove(f"{export_directory}/precision_recall_curve.png")

    fs = 15

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'AUC = {average_precision:.2f}', color="#3668B2")
    plt.scatter(recall[f1_idx], precision[f1_idx], color="#C44747",
                label=f"Max F1 = {f1_scores[f1_idx]:.2f} (at thr={f1_threshold:.2f})", zorder=5, s=30)
    print(f"Precision at max F1: {precision[f1_idx]:.2f}, Recall at max F1: {recall[f1_idx]:.2f}")
    plt.plot([recall[f1_idx], recall[f1_idx]], [precision[0], precision[f1_idx]],
             color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.plot([0, recall[f1_idx]], [precision[f1_idx], precision[f1_idx]],
             color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.xlabel('Recall', fontsize=fs)
    plt.ylabel('Precision', fontsize=fs)
    plt.title('Precision-Recall Curve', fontsize=fs)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=fs-2)
    plt.savefig(f"{export_directory}/precision_recall_curve.png", dpi=300)
    plt.close()

    print(f"Precision-recall curve saved as {export_directory}/precision_recall_curve.png")

    return precision, recall, average_precision

def save_evaluation_results(export_directory, fpr, tpr, roc_auc, precision, recall, average_precision, feature_names, feature_importance):
    """Saves the evaluation results to a JSON file.

    :param export_directory: Directory to save the evaluation results.
    :param fpr: False positive rates.
    :param tpr: True positive rates.
    :param roc_auc: Area under the ROC curve.
    :param precision: Precision values.
    :param recall: Recall values.
    :param average_precision: Average precision score.
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
            "recall": recall.tolist(),
            "average_precision": average_precision
        },
        "feature_importance": {
            "feature_names": feature_names,
            "importance": feature_importance
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

    #print(f'Prob: {y_prob[0:10]} - True: {y_test[0:10]}')
    # how many correct predictions with threshold 0.5 and 0.31

    y_pred_05 = (y_prob >= 0.5).astype(int)
    #y_pred_031 = (y_prob >= 0.31).astype(int)
    accuracy_05 = np.mean(y_pred_05 == y_test)
    #accuracy_031 = np.mean(y_pred_031 == y_test)
    print(f'Accuracy at threshold 0.5: {accuracy_05:.4f}')
    #print(f'Accuracy at threshold 0.31: {accuracy_031:.4f}')
    print(f'Size of test set: {len(y_test)}, Positives: {np.sum(y_test)}, Negatives: {len(y_test) - np.sum(y_test)}')
    print(f'Predicted positives at threshold 0.5: {np.sum(y_pred_05)}, Predicted negatives at threshold 0.5: {len(y_test) - np.sum(y_pred_05)}')
    #print(f'Predicted positives at threshold 0.31: {np.sum(y_pred_031)}, Predicted negatives at threshold 0.31: {len(y_test) - np.sum(y_pred_031)}')
    # print confusion matrix
    from sklearn.metrics import confusion_matrix
    cm_05 = confusion_matrix(y_test, y_pred_05)
    #cm_031 = confusion_matrix(y_test, y_pred_031)
    print(f'Confusion matrix at threshold 0.5:\n{cm_05}')
    #print(f'Confusion matrix at threshold 0.31:\n{cm_031}')

    # Generate and print ROC curve
    fpr, tpr, roc_auc = plot_roc_curve(y_test, y_prob, export_directory)

    # Print precision-recall curve
    precision, recall, average_precision = plot_precision_recall_curve(y_test, y_prob, export_directory)

    # Print feature importance
    if hasattr(clf, 'feature_importances_'):
        sorted_feature_list, sorted_feature_importance = plot_feature_importance(clf, feature_list, export_directory)
        sorted_feature_importance = sorted_feature_importance.tolist()
    else:
        sorted_feature_list = feature_list
        sorted_feature_importance = []

    # Save evaluation results
    save_evaluation_results(export_directory, fpr, tpr, roc_auc, precision, recall, average_precision, sorted_feature_list, sorted_feature_importance)

    hyper_parameters = clf.get_params()
    print(hyper_parameters)
    with open(f"{export_directory}/hyperparameters.json", 'w') as f:
        json.dump(hyper_parameters, f, indent=4)
    print(f"Hyperparameters saved as {export_directory}/hyperparameters.json")