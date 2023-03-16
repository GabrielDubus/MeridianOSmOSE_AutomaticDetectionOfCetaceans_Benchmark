import pandas as pd
from pathlib import Path
from ketos.data_handling.data_handling import parse_datetime
from ketos.audio.waveform import get_duration
from ketos.data_handling.data_handling import find_files
from random import sample
import os
import numpy as np

def file_duration_table(path, datetime_format=None, num=None, exclude_subdir=None):
    """ Create file duration table.

        Args:
            path: str
                Path to folder with audio files (\*.wav)
            datetime_format: str
                String defining the date-time format. 
                Example: %d_%m_%Y* would capture "14_3_1999.txt".
                See https://pypi.org/project/datetime-glob/ for a list of valid directives.
                If specified, the method will attempt to parse the datetime information from the filename.
            num: int
                Randomly sample a number of files
            exclude_subdir: str
                Exclude subdir from the search 

        Returns:
            df: pandas DataFrame
                File duration table. Columns: filename, duration, (datetime)
    """
    paths = find_files(path=path, substr=['.wav', '.WAV', '.flac', '.FLAC'], search_subdirs=True, return_path=True)

    if exclude_subdir is not None:
        paths = [path for path in paths if exclude_subdir not in path]

    if num is not None:
        paths = sample(paths, num)

    durations = get_duration([os.path.join(path,p) for p in paths])
    df = pd.DataFrame({'filename':paths, 'duration':durations})
    if datetime_format is None:
        return df

    df['datetime'] = df.apply(lambda x: parse_datetime(os.path.basename(x.filename), fmt=datetime_format), axis=1)
    return df

def export(detections, save_to, mode='w', export_mode=None, audio_representation=None):
    if export_mode == 'raven':
        export_to_raven(detections, save_to, mode=mode, audio_representation=audio_representation)
    else:
        export_detections(detections, save_to, mode=mode)

def export_detections(detections, save_to, mode='w'):
    """ Save the detections to a csv file

        Args:
            detections: tuple
                Detections. Each entry is a tuple of the form 
                (filename, offset, duration, score)
            save_to: string
                The path to the .csv file where the detections will be saved.
                Example: "/home/user/detections.csv"
            mode: I/O file mode
    """
    if len(detections) == 0 and mode != 'w': 
        return
    
    detections['label'] = 1
    
    include_header = True
    if mode == 'a':
        include_header = False
    detections.to_csv(save_to, mode=mode, index=False, header=include_header)

def export_to_raven(detections, save_to, audio_representation, mode='w'):
    """ Save the detections to a csv file

        Args:
            detections: tuple
                Detections. Each entry is a tuple of the form 
                (filename, offset, duration, score)
            save_to: string
                The path to the .csv file where the detections will be saved.
                Example: "/home/user/detections.csv"
            mode: I/O file mode
    """
    if len(detections) == 0 and mode != 'w': 
        return

    max_freq = None
    min_freq = None
    if 'freq_max' in audio_representation:
        max_freq = audio_representation['freq_max']
    if 'freq_min' in audio_representation:
        min_freq = audio_representation['freq_min']

    raven_df = pd.DataFrame()
    raven_df['Selection'] = list(range(len(detections)))
    raven_df['View'] = 'Spectrogram'
    raven_df['Channel'] = 0
    raven_df["Begin Time (s)"] = detections['start']
    raven_df["End Time (s)"] = detections['end']
    raven_df['Delta Time (s)'] = detections['end'] - detections['start']
    raven_df['File Offset (s)'] = detections['start']
    raven_df['Low Freq (Hz)'] = min_freq
    raven_df['High Freq (Hz)'] = max_freq
    raven_df["Begin File"] = [Path(detection).name for detection in detections['filename']]
    raven_df["Begin Path"] = [str(Path(detection).parent) for detection in detections['filename']]
    raven_df['Species ID'] = 1
    raven_df["Comments"] = detections['scores']
    
    include_header = True
    if mode == 'a':
        include_header = False
    raven_df.to_csv(save_to, mode=mode, index=False, header=include_header)    

    
def export_to_hallo(output_path, detections, freq_min=0, freq_max=2500, mode='w'):
    """ Save DataFrame to Hallo annotation-App format.

        Args:
            output_path: str
                Path to output csv file
            detections: list(tuple)
                Detections. Each entry is a tuple of the form 
                (filename, offset, duration, score)
            freq_min,freq_max: float
                Minimum and maximum frequency (Hz) of the detections
            channel: int
                Audio channel (1,2,3,...)
    """

    # define columns of output csv file
    data = {"filename": [], 
             "start": [],
             "end": [],
             "freq_min": [],
             "freq_max": [],
             "offset": [],
             "sound_id_species": [],
             "kw_ecotype": [],
             "pod": [],
             "call_type": [],
             "comments": [], 
           }

    # loop over detections
    for det in detections:
        if det[1] < 10:
            continue
        filename = os.path.basename(det[0])
        start = det[1]
        duration = det[2]
        score = det[3]

        data["filename"].append(filename)
        data["start"].append(start)
        data["end"].append(start + duration)
        data["freq_min"].append(freq_min)
        data["freq_max"].append(freq_max)
        data["offset"].append(0)
        data["sound_id_species"].append('')
        data["kw_ecotype"].append('')
        data["pod"].append('')
        data["call_type"].append('')
        data["comments"].append("")
         
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, sep=';', mode=mode)

def merge_detections(detection_df):
    """ Merge overlapping or adjacent detections.

        The score of the merged detection is computed as the average of the individual detection scores.

        Note: The detections are assumed to be sorted by start time in chronological order.

        Args:
            detection_df: pandas DataFrame
                Dataframe with detections
        
        Returns:
            merged: pandas DataFrame
                DataFrame with the merged detections
    """
    detections = detection_df.to_dict('records')

    if len(detections) <= 1:
        return detection_df
    
    merged_detections = [detections[0]]

    for i in range(1,len(detections)):
        # detections do not overlap, nor are they adjacent
        if detections[i]['start'] > merged_detections[-1]['end'] or detections[i]['filename'] != merged_detections[-1]['filename']:
            merged_detections.append(detections[i])
        # detections overlap, or adjacent to one another
        else:
            avg_score = 0.5 * (detections[i]['scores'] + merged_detections[-1]['scores']) 
            merged_detection = {'filename': detections[i]['filename'], 'start': merged_detections[-1]['start'], 'end': detections[i]['end'], 'scores': avg_score}
            merged_detections[-1] = merged_detection #replace     

        return pd.DataFrame(merged_detections)