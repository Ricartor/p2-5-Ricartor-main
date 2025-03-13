import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    tp = sum((y_pred_mapped == 1) & (y_true_mapped == 1))
    tn = sum((y_pred_mapped == 0) & (y_true_mapped == 0))
    fp = sum((y_pred_mapped == 1) & (y_true_mapped == 0))
    fn = sum((y_pred_mapped == 0) & (y_true_mapped == 1))

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0

    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'Confusion Matrix': [tn, fp, fn, tp],
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1
    }

def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class (1).
        positive_label: The label considered as the positive class.
        n_bins (int, optional): Number of equally spaced bins to use for calibration. Defaults to 10.

    Returns:
        None: This function plots the calibration curve and does not return any value.

    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    true_proportions = np.zeros(n_bins)

    for i in range(n_bins):
        indices = (y_probs >= bins[i]) & (y_probs < bins[i+1])
        if np.sum(indices) > 0:
            true_proportions[i] = np.mean(y_true_mapped[indices])

    plt.plot(bin_centers, true_proportions, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.show()

def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class.
        positive_label: The label considered as the positive class.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10.

    Returns:
        None: This function plots the histograms and does not return any value.

    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    plt.figure(figsize=(12, 6))

    # Histogram for positive class
    plt.subplot(1, 2, 2)
    plt.hist(y_probs[y_true_mapped == 1], bins=n_bins, color='green', alpha=0.7)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Histogram (Positive Class)')

    # Histogram for negative class
    plt.subplot(1, 2, 1)
    plt.hist(y_probs[y_true_mapped == 0], bins=n_bins, color='red', alpha=0.7)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Histogram (Negative Class)')

    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class (1).
        positive_label: The label considered as the positive class.

    Returns:
        None: This function plots the ROC curve.

    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    thresholds = np.linspace(0, 1, 11)
    tpr = []  # True Positive Rate
    fpr = []  # False Positive Rate

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        tp = np.sum((y_true_mapped == 1) & (y_pred == 1))
        fp = np.sum((y_true_mapped == 0) & (y_pred == 1))
        fn = np.sum((y_true_mapped == 1) & (y_pred == 0))
        tn = np.sum((y_true_mapped == 0) & (y_pred == 0))

        tpr.append(tp / (tp + fn) if tp + fn != 0 else 0)
        fpr.append(fp / (fp + tn) if fp + tn != 0 else 0)

    tpr.append(0)
    fpr.append(0)

    plt.plot(fpr, tpr, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

def classification_report(y_true, y_probs, positive_label, threshold=0.5, n_bins=10):
    """
    Create a classification report using the auxiliary functions developed during Lab2_1

    Args:
        y_true (array-like): True labels of the data.
        y_probs (array-like): Predicted probabilities for the positive class.
        positive_label: The label considered as the positive class.
        threshold (float): Threshold to transform probabilities to predictions. Defaults to 0.5.
        n_bins (int, optional): Number of bins for the histograms and equally spaced 
                                bins to use for calibration. Defaults to 10.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)

    """
    plot_calibration_curve(y_true, y_probs, positive_label, n_bins)
    plot_probability_histograms(y_true, y_probs, positive_label, n_bins)
    plot_roc_curve(y_true, y_probs, positive_label)
    return evaluate_classification_metrics(y_true, (y_probs > threshold).astype(int), positive_label)