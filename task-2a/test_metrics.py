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
predicted = pd.DataFrame(data)


# I think the above is the minimum field necessary to calculate the metrics. Tell me what you think.

total_hours = 10

############################


# Calculate precision, recall, and F1 score manually
def calculate_metrics(ground_truth, predicted, threshold):
    tp = fp = fn = 0

    # Filter positive scores based on threshold
    predicted_filtered = predicted[predicted['score'] >= threshold]

    # We are now going to calculate the TP and FN. Note that this calculation covers instances where we have one prediction that overlaps with
    #  multiple ground truths. In this case, the TP will be incremented for each ground truth
    # We are also considering Multiple preditcions for one ground truth, in this case, only one of the predictions will increment the TP.
    # Another assumption made here is that even if only 1% of the prediction falls within the ground truth it will count as a TP

    # Loop through each ground truth entry
    for _, gt_row in ground_truth.iterrows():
        gt_interval = pd.Interval(gt_row['start'], gt_row['end'])
        gt_filename = gt_row['wav_filename']

        # Find predicted entries with the same filename as the ground truth entry
        matching_predicted = predicted_filtered[predicted_filtered['wav_filename'] == gt_filename]

        # Check if there's any overlap between the ground truth entry and any of the matching predicted entries
        overlap_found = False
        for _, predicted_row in matching_predicted.iterrows():
            predicted_interval = pd.Interval(predicted_row['start'], predicted_row['end'])
            if gt_interval.overlaps(predicted_interval):
                overlap_found = True
                break
        
        # If an overlap is found, increment TP, otherwise increment FN
        if overlap_found:
            tp += 1
        else:
            fn += 1
    
    # Calculate FP:
    # Loop through each predicted filtered entry
    for _, predicted_row in predicted_filtered.iterrows():
        predicted_interval = pd.Interval(predicted_row['start'], predicted_row['end'])
        predicted_filename = predicted_row['wav_filename']

        # Find ground_truth entries with the same filename as the predicted filtered entry
        matching_ground_truths = ground_truth[ground_truth['wav_filename'] == predicted_filename]

        # Check if there's any overlap between the predicted filtered entry and any of the matching ground_truth entries
        overlap_found = False
        for _, gt_row in matching_ground_truths.iterrows():
            gt_interval = pd.Interval(gt_row['start'], gt_row['end'])
            if gt_interval.overlaps(predicted_interval):
                overlap_found = True
                break
        
        # If no overlap is found, increment FP
        if not overlap_found:
            fp += 1
    

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    fpr_per_hour = fp / total_hours

    return precision, recall, f1, fpr_per_hour

thresholds = np.linspace(0, 1, 101)
metrics = {
    "threshold": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "fpr_h": []
}

# Iterate through different thresholds to calculate metrics
for threshold in thresholds:


    precision, recall, f1, fpr_per_hour = calculate_metrics(ground_truth, predicted, threshold)
    metrics["threshold"].append(threshold)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1"].append(f1)
    metrics['fpr_h'].append(fpr_per_hour)


# Print the results for each threshold
# for threshold, precision, recall, f1, fp_per_hour in zip(metrics['threshold'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['fpr_h']):
#     print(f'Threshold: {threshold:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, FP per hour: {fp_per_hour:.2f}')


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

def plot_fpr_h_threshold_curve(metrics):
    fpr_hs = metrics['fpr_h']
    thresholds = metrics['threshold']

    plt.plot(thresholds, fpr_hs, marker='.')
    plt.xlim([0, 1])
    plt.ylim([0, max(fpr_hs) * 1.1])  # Give some extra space on the top of the plot
    plt.xlabel('Threshold')
    plt.ylabel('False Positives per Hour')
    plt.title('False Positives per Hour vs Threshold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot the Precision-Recall curve
plot_precision_recall_curve(metrics)

# Plot the F1 Score-Threshold curve
plot_f1_score_threshold_curve(metrics)

# Plot the Threshold Curves
plot_fpr_h_threshold_curve(metrics)