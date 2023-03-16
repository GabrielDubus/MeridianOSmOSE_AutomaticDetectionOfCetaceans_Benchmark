#!/usr/bin/python


import os
import numpy as np
import pandas as pd
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.neural_networks.dev_utils.detection import merge_overlapping_detections
from narw_detector.utils.train_model_utils import import_nn_interface, which_nn_interface
from narw_detector.utils.utils import export, find_files, merge_detections as merge
from pathlib import Path
from tqdm import trange
import warnings
    
def run_model(model, audio_folder, file_list=None, threshold=0.5, step_size=None, batch_size=32, output=None, overwrite=True, save_format=None, merge_detections=False):

    # Import the neural-network interface class
    nn_recipe, _, module_path = which_nn_interface(model)
    nn_interface = import_nn_interface(name=nn_recipe['interface'], module_path=module_path)

    # load model and audio representation(s)
    model, audio_representation = nn_interface.load(model, load_audio_repr=True)
    audio_representation = [list(a.values())[0] for a in audio_representation][0]
    
    # if step size was not specified, set it equal to the window size
    if step_size is None:
        step_size = audio_representation['duration']

    audio_files = [os.path.join(audio_folder, f) for f in find_files(audio_folder, '.wav')]

    # filter files based on a list
    if file_list is not None:
        file_list = pd.read_csv(file_list, header=None)[0].values.tolist()
        audio_files = [row for row in audio_files if row.rsplit(os.sep,1)[1] in file_list] # filtering all the files paths to only process the ones we specified in file_list

    if output is None:
        output = Path('detections', 'detections.csv').resolve()
    else:
        output = Path(output).resolve()

    output.parent.mkdir(parents=True, exist_ok=True)

    mode = 'a'
    if overwrite:
        mode = 'w'
    else:
        # if not overwrite then check the detection file to see if we are resuming from were the processing stopped.
        # and filter the files that were already processed
        detected_df = pd.read_csv(output)
        processed_files = pd.unique(detected_df['filename'])
        if len(processed_files) > 0:
            # Getting the index of the last processed file 
            idx = len(processed_files) - 1
            # Filtering the list of audio files to include only the files not in the processed files + this last file
            # processing this last file again is necessary because we do not know at which point in the file the processing stopped
            audio_files = audio_files[idx:]


    # initialize audio loader
    # The audio loader will segment an audio recording into segments of size "duration"
    # This is a generator that yields each segment as a spectrogram
    loader = AudioFrameLoader(duration=audio_representation["duration"], pad=False, step=step_size, batch_size=1,
            filename=audio_files, representation=audio_representation['type'], representation_params=audio_representation)
    
    n_batches = int(np.ceil(loader.num() / batch_size))

    # Boolean to flag if something was deteted
    detected = False
    for batch_id in trange(n_batches):
        num_samples = min(batch_size, loader.num() - batch_size * batch_id) 

        # batch data will contain tuples (filename, start, duration)
        batch_data = {"filename": [], "start": [], 'end': []}
        data = []
        warnings.simplefilter('module', RuntimeWarning)
        for _ in range(num_samples):
            spec = next(loader)
            data.append(spec.data.data) 
            batch_data['filename'].append(spec.filename)
            batch_data['start'].append(spec.offset)
            batch_data['end'].append(spec.offset + spec.duration())
 
        # transform batch data to be a numpy array with shape (batch_size, time_bins, freq_bins)
        data = np.stack(data, axis=0)
        
        # the run on batch method will apply the transform the data following to match the input (batch_size, num_classes)
        batch_predictions = model.run_on_batch(data, transform_input=True, return_raw_output=True) 
        
        # getting the scores of label 1
        scores = batch_predictions[:,1]

        df = pd.DataFrame(batch_data)
        df['scores'] = scores

        # Filtering for the threshold
        batch_detections = df[df["scores"] >= threshold].reset_index(drop=True)
        
        if len(df) > 0:
            detected = True

        if merge_detections:
            batch_detections = merge(batch_detections)

        if batch_id == 0 and overwrite == False and len(processed_files) > 0:
            # Filtering out all the detections from the last filename
            detected_df = detected_df[detected_df['filename'] != processed_files[-1]].reset_index(drop=True)
            detected_df.to_csv(output, mode='w', index=False, header=True)

        # save batch detections 
        export(batch_detections, output, mode=mode, export_mode=save_format, audio_representation=audio_representation)

        # We are writing after every batch, so even if overwrite was True, at this point we need to append to the file
        mode = "a"

    if detected:
        print(f'Detections saved to: {output}')

    else:
        print("The model did not detect any vocalizations")
    

def main():
    import argparse

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Saved model (*.kt)')
    parser.add_argument('audio_folder', type=str, help='Folder with audio files')
    parser.add_argument('--file_list', default=None, type=str, help='A csv or .txt file where each row (or line) is the name of a file to detect within the audio folder. By default, all files will be processed.')
    parser.add_argument('--threshold', default=0.5, type=float, help='Detection threshold (must be between 0 and 1)')
    parser.add_argument('--step_size', default=None, type=float, help='Step size in seconds. If not specified, the '\
        'step size is set equal to the duration of the audio representation.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--output', default=None, type=str, help='Detection output .csv. For instance: detections/my_detections.csv')
    parser.add_argument('--save_format', default=None, type=str, help='To save in raven format specify --save_format raven')
    parser.add_argument('--merge_detections', default=False, type=boolean_string, help='Merge adjacent or overlapping detections.')
    parser.add_argument('--overwrite', default=True, type=boolean_string, help='Overwrites the detections, otherwise appends to it. Allows resuming processing files.')

    args = parser.parse_args()
    run_model(**vars(args))

if __name__ == "__main__":
    main()
