import numpy as np
import pandas as pd

# Creating ground truth DataFrame
data = {
    'wav_filename': ['file1.wav', 'file1.wav', 'file2.wav', 'file3.wav', 'file3.wav'],
    'start': [2.0, 5.0, 8.0, 12.0, 15.0],
    'end': [3.0, 6.0, 9.0, 13.0, 16.0]
}
ground_truth = pd.DataFrame(data)

# Creating positive_scores DataFrame
data = {
    'wav_filename': ['file1.wav', 'file1.wav', 'file2.wav', 'file2.wav', 'file3.wav', 'file3.wav', 'file3.wav'],
    'start': [2.5, 5.7, 8.3, 10.1, 12.6, 15.8, 18.0],
    'end': [3.5, 6.7, 9.3, 11.1, 13.6, 16.8, 19.0],
    'score': [0.9, 0.2, 0.6, 0.8, 0.3, 0.95, 0.1]
}
positive_scores = pd.DataFrame(data)

total_hours = 10

############################

# Check if the predicted interval overlaps with any ground truth interval for the same WAV file
def is_positive(filename, start_pred, end_pred, ground_truth):
    file_ground_truth = ground_truth[ground_truth['wav_filename'] == filename]
    predicted_interval = pd.Interval(start_pred, end_pred)
    for _, row in file_ground_truth.iterrows():
        ground_truth_interval = pd.Interval(row['start'], row['end'])
        if ground_truth_interval.overlaps(predicted_interval):
            return True
    return False


# Calculate precision, recall, and F1 score manually
def calculate_metrics(y_true, y_pred):
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 0 and yp == 0:
            tn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0
    tpr = tp / (tp + fn) if tp + fn > 0 else 0
    fp_per_hour = fp / total_hours

    return precision, recall, f1, fpr, tpr, fp_per_hour

thresholds = np.linspace(0, 1, 101)
metrics = {
    "threshold": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "fpr": [],
    "tpr": [],
    "fpr_h": []
}

# Iterate through different thresholds to calculate metrics
for threshold in thresholds:
    y_true = [1 if is_positive(row['wav_filename'], row['start'], row['end'], ground_truth) else 0 for _, row in positive_scores.iterrows()]
    y_pred = [1 if row['score'] >= threshold else 0 for _, row in positive_scores.iterrows()]

    precision, recall, f1, fpr, tpr, fp_per_hour = calculate_metrics(y_true, y_pred)
    metrics["threshold"].append(threshold)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1"].append(f1)
    metrics['fpr_h'].append(fp_per_hour)
    metrics["fpr"].append(fpr)
    metrics["tpr"].append(tpr)

# Print the results for each threshold
for threshold, precision, recall, f1, fp_per_hour in zip(metrics['threshold'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['fpr_h']):
    
    print(f'Threshold: {threshold:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, FP per hour: {fp_per_hour:.2f}')


import matplotlib.pyplot as plt

def plot_precision_recall_curve(metrics):
    precisions = metrics['precision']
    recalls = metrics['recall']
    
    plt.plot(recalls, precisions, marker='.')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_f1_score_threshold_curve(metrics):
    f1_scores = metrics['f1']
    thresholds = metrics['threshold']
    
    plt.plot(thresholds, f1_scores, marker='.')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score-Threshold Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(metrics):
    fpr_list = metrics['fpr']
    tpr_list = metrics['tpr']

    plt.plot(fpr_list, tpr_list, marker='.')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the Precision-Recall curve
plot_precision_recall_curve(metrics)

# Plot the F1 Score-Threshold curve
plot_f1_score_threshold_curve(metrics)

# Plot the ROC curve
plot_roc_curve(metrics)
