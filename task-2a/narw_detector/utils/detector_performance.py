import pandas as pd
import os



def model_performance(detector_csv, actual_csv):
    actual_csv = actual_csv
    df_predicted = pd.read_csv(detector_csv) # start, duration

    df_predicted["filename"] = df_predicted["filename"].apply(lambda x: os.path.basename(x))

    df_actual = pd.read_csv(actual_csv) # start, end


    TP = 0
    FP = 0
    FN = 0

    cnt = 0
    list_of_FP = pd.DataFrame()
    for index,row in df_predicted.iterrows(): 
        if row['start'] < 10:
            continue
        df = df_actual[df_actual['filename'] == row['filename']]

        predicted_interval = pd.Interval(row['start'], row['end'])

        predicted_actual_match = False
        for i,df_row in df.iterrows():
            actual_interval = pd.Interval(df_row['start'], df_row['end']) 
            
            if actual_interval.overlaps(predicted_interval) | predicted_interval.overlaps(actual_interval):
                predicted_actual_match = True
                break
        
        if predicted_actual_match:
            TP += 1
        else:
            FP += 1
            # list_of_FP = pd.concat([list_of_FP, row])

    # list_of_FP.to_csv("misc/list_of_FP.csv", index=False)
    FN = len(df_actual)-TP
    if FN < 0:
        FN = 0
    print(f"True Positives: {TP}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    FPR = FP / 18.5
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"FPR: {FPR}")

if __name__ == "__main__":
    import argparse
    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('detector_csv', type=str, help='Path to the csv with the predictions.')
    parser.add_argument('actual_csv', type=str, help='Path to the csv with the annotations.')
    

    args = parser.parse_args()
    model_performance(**vars(args))

