import numpy as np
import tensorflow as tf
import pandas as pd
import csv
from pathlib import Path
import ketos.data_handling.database_interface as dbi
import os
from ketos.data_handling.data_feeding import BatchGenerator, JointBatchGen
from narw_detector.utils.train_model_utils import import_nn_interface, which_nn_interface
from matplotlib import pyplot as plt

def compute_detections(labels, scores, threshold=0.5):
    #The following will calculate predictions but only works for binary problems
    predictions = np.where(scores >= threshold, 1, 0)
    
    TP = tf.math.count_nonzero(predictions * labels).numpy()
    TN = tf.math.count_nonzero((predictions - 1) * (labels - 1)).numpy()
    FP = tf.math.count_nonzero(predictions * (labels - 1)).numpy()
    FN = tf.math.count_nonzero((predictions - 1) * labels).numpy()

    return predictions, TP, TN, FP, FN


def process_hdf5_db(model, nn_interface, hdf5_db, table_name=None, output_dir='.', threshold=0.5,  batch_size=32):
    db = dbi.open_file(hdf5_db, 'r')

    if table_name is None:
        table_name = "/"
    table = dbi.open_table(db, table_name)
    
    gens = []
    batch_size = int(batch_size / sum(1 for _ in db.walk_nodes(table, "Table")))
    for group in db.walk_nodes(table, "Table"):
        generator = BatchGenerator(batch_size=batch_size,
                            data_table=group, map_labels=False,
                            output_transform_func=nn_interface.transform_batch,
                            shuffle=False, refresh_on_epoch_end=False, x_field="data", return_batch_ids=True)
        gens.append(generator)
        

    # join generators
    gen = JointBatchGen(gens, n_batches="min", shuffle_batch=False, map_labels=False, reset_generators=False, return_batch_ids=True)

    print('\nPredicting on the table data...')

    scores = []
    labels = []
    
    for batch_id in range(gen.n_batches):
        hdf5_ids, batch_X, batch_Y = next(gen)
        batch_labels = np.argmax(batch_Y, axis=1) #convert from 1-hot encoding
        batch_scores = model.model.predict_on_batch(batch_X)[:,1] # will return the scores for just one class (with label 1)

        scores.extend(batch_scores)
        labels.extend(batch_labels)
    
    # Converting list to np array
    labels = np.array(labels)
    scores = np.array(scores)
    predicted, TP, TN, FP, FN = compute_detections(labels, scores, threshold)
    
    print(f'\nSaving detections output to {output_dir}/')
    df_group = pd.DataFrame()
    for group in db.walk_nodes(table, "Table"):
        df = pd.DataFrame({'id':group[:]['id'], 'filename':group[:]['filename']})
        df_group = pd.concat([df_group, df], ignore_index=True)
    df_group['label'] = labels[:]
    df_group['predicted'] = predicted[:]
    df_group['score'] = scores[:]
    df_group.to_csv(os.path.join(os.getcwd(), output_dir, "classifcations.csv"), mode='w', index=False)

    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    FPP = FP / (TN + FP)
    confusion_matrix = [[TP, FN], [FP, TN]]
    print(f'\nPrecision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('\nConfusionMatrix:')
    print('\n[TP, FN]')
    print('[FP, TN]')
    print(f'{confusion_matrix[0]}')
    print(f'{confusion_matrix[1]}')

    print(f"\nSaving metrics to {output_dir}/")

    # Saving precision recall and F1 Score for the defined thrshold
    metrics = {'Precision': [precision], 'Recall': [recall], "F1 Score": [f1]}
    metrics_df = pd.DataFrame(data=metrics)

    metrics_df.to_csv(os.path.join(os.getcwd(), output_dir, "metrics.csv"), mode='w', index=False)
    print("\n Ploting graphs")

    # Appending a confusion matrix to the file
    row1 = ["Confusion Matrix", "Predicted"]
    row2 = ["Actual", "NARW", "Background Noise"]
    row3 = ["NARW", TP, FN]
    row4 = ["Background Noise", FP, TN]
    with open(os.path.join(os.getcwd(), output_dir, "metrics.csv"), 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(row1)
        writer.writerow(row2)
        writer.writerow(row3)
        writer.writerow(row4)

    plot_test_curves(labels, scores, output_dir)

    db.close()    


def plot_test_curves(labels, scores, output_dir):
    increment = 0.05
    thresholds = np.arange(0.05, 1, increment) # Will generate thresholds between 0.05 and 0.95 with 0.05 increments
    
    recalls = []
    precisions = []
    FPPs = []
    for threshold in thresholds:      
        _, TP, TN, FP, FN = compute_detections(labels, scores, threshold)
        precisions.append(TP / (TP + FP))
        recalls.append(TP / (TP + FN))
        FPPs.append(FP / (TN + FP))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
    ax.plot(recalls, FPPs)

    ax.grid(linewidth=1.5, alpha=0.8)
    ax.minorticks_on()
    ax.xaxis.grid(which='minor', alpha=0.6, linewidth=0.5)
    ax.yaxis.grid(which='minor', alpha=0.6, linewidth=0.5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('False Positive Probability')
    # ax.set_ylim(0., .1)
    # ax.set_xlim(0., 1.0)
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "recallxFPR.png")
    plt.savefig(fig_path)
    print(f' Figure saved to: {fig_path}')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, precisions, label="Precision")

    ax.grid(linewidth=1.5, alpha=0.8)
    ax.minorticks_on()
    ax.xaxis.grid(which='minor', alpha=0.6, linewidth=0.5)
    ax.yaxis.grid(which='minor', alpha=0.6, linewidth=0.5)
    
    ax.set_xlabel('Detection Threshold')
    ax.set_ylabel('Recall', color="#1f77b4")
    ax.set_xlim(0.0, 1.0)
    ax2 = ax.secondary_yaxis("right")
    ax2.minorticks_on()
    ax2.set_ylabel('Precision', color="#ff7f0e")
    ax2.set_ylim(0.0,1.0)
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "Detection_thresholds.png")
    plt.savefig(fig_path)
    print(f' Figure saved to: {fig_path}')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
    ax.plot(recalls, precisions)

    ax.grid(linewidth=1.5, alpha=0.8)
    ax.minorticks_on()
    ax.xaxis.grid(which='minor', alpha=0.6, linewidth=0.5)
    ax.yaxis.grid(which='minor', alpha=0.6, linewidth=0.5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    # ax.set_ylim(0., .1)
    # ax.set_xlim(0., 1.0)
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "precXrec.png")
    plt.savefig(fig_path)
    print(f' Figure saved to: {fig_path}')



def test_model(model, hdf5_db, table_name=None, batch_size=32, output_dir=None, threshold=0.5):
    # Import the neural-network interface class
    nn_recipe, _, module_path = which_nn_interface(model)
    nn_interface = import_nn_interface(name=nn_recipe['interface'], module_path=module_path)

    # load model and audio representation(s)
    model = nn_interface.load(model, load_audio_repr=False)
    
    if output_dir is None:
        output_dir = 'test_results'
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    process_hdf5_db(model, nn_interface, hdf5_db, table_name, batch_size=batch_size, output_dir=output_dir, threshold=threshold)

def main():
    import argparse
    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Saved model (*.kt)')
    parser.add_argument('hdf5_db', type=str, help="hdf5 database path")
    parser.add_argument('--table_name', default=None, type=str, help="Table name within the database where the test data is stored. Must start with a foward slash. For instance '/test'. If not given, the root '/' path will be used")
    parser.add_argument('--threshold', default=0.5, type=float, help='Detection threshold (must be between 0 and 1)')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--output_dir', default=None, type=str, help='The folder where the results will be saved.')
    
    args = parser.parse_args()
    test_model(**vars(args))

if __name__ == "__main__":
    main()
