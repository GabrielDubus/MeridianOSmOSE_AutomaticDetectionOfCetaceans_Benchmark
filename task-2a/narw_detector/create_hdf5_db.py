import numpy as np
import os
import copy
import pandas as pd
from pathlib import Path
from ketos.data_handling import selection_table as sl
from ketos.data_handling.parsing import load_audio_representation
import ketos.data_handling.database_interface as dbi
from narw_detector.utils.utils import file_duration_table


def create_db(data_dir, audio_representation, annotations=None, annotation_step=0, step_min_overlap=0.5, labels=None, output=None, table_name=None, random_selections=None, overwrite=False, seed=None):
    #initialiaze random seed
    if seed is not None:
        np.random.seed(seed)

    if random_selections is None and annotations is None:
        raise Exception("Missing value: Either annotations or random_selection must be defined.") 

    selections = {}
    
    #load the audio representation. We are currently only allowing 1
    config = load_audio_representation(audio_representation)
    config = config[list(config.keys())[0]]

    annots = None
    if annotations is not None: # if an annotation table is given
        annots = pd.read_csv(annotations)

        label_dict = None
        if labels is None:
            labels = annots.label.unique().tolist() # For each unique label
        else:
            label_dict = copy.copy(labels) # Copying the content of labels to use it later since we are going to modify the labels variable now
            labels = list(labels.keys())
      
        annots = sl.standardize(table=annots, trim_table=True, labels=labels, start_labels_at_1=False) #standardize to ketos format and remove extra columns
        
        # this is a work around on the issue that we cant tell the program which labels to use. It will always map labels to 0,1,3,4...
        # This owrkaround will then map the automatic created labels back to what the user wants
        if label_dict is not None:
            for key in label_dict:
                #this is a hack because the standardize function convert keys with numeric string to an int, while label dict will maintain it as string ###
                standardize_annot_key = key
                if key.isdigit():
                    standardize_annot_key = int(standardize_annot_key)

                ### annots.attrs['label_dict'] contain the mapping done by the standardize function
                ### label_dict[key] contains the value we actually want for that label
                annots['label'].replace(annots.attrs['label_dict'][standardize_annot_key], label_dict[key], inplace=True)

        labels = annots.label.unique().tolist() # get the actual list of labels after all the processing
        # removing label -1
        labels = [label for label in labels if label != -1]
        if 'start' in annots.columns and 'end' in annots.columns: # Checking if start and end times are in the dataframe
            for label in labels:
                selections[label] = sl.select(annotations=annots, length=config['duration'], step=annotation_step, min_overlap=step_min_overlap, center=False, label=[label]) #create the selections
        else: # if not, than the annotations are already the selections
            for label in labels:
                selections[label] = annots.loc[annots['label'] == label]

    # random_selections is a list where the first index is the number of samples to generate and the second index is the label to assign to the generations
    if random_selections is not None: 
        print(f'\nGenerating {random_selections[0]} samples with label {random_selections[1]}...')
        files = file_duration_table(data_dir, num=None)
        rando = sl.create_rndm_selections(files=files, length=config['duration'], annotations=annots, num=random_selections[0], label=random_selections[1])
        del rando['duration'] # create_rndm selections returns the duration which we dont need. So lets delete it

        if labels is None:
            labels = []
            
        if random_selections[1] in labels: 
            # if the random selection label already exists in the selections, concatenate the generatiosn with the selections that already exist
            selections[random_selections[1]] = pd.concat([selections[random_selections[1]], rando], ignore_index=False) # concatenating the generated random selections with the existings selections
        else:
            # if the random selections label did not yet exist in the selections, add it to the list of labels
            labels.append(random_selections[1])
            selections[random_selections[1]] = rando

    if output is None:
        output = os.path.join('db', 'narw_db.h5')

    if not os.path.isabs(output): 
        output = os.path.join(os.getcwd(), output)
    Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True) #creating dir if it doesnt exist

    if overwrite and os.path.exists(output): 
        os.remove(output)

    print('\nCreating db...')
    for label in labels:
        print(f'\nAdding data with label {label} to table {table_name}...')
        dbi.create_database(output_file=output, data_dir=data_dir, dataset_name=table_name, selections=selections[label], audio_repres=config)

def main():
    import argparse

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split('=')
                if value.isdigit():
                    value = int(value)
                getattr(namespace, self.dest)[key] = value

    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the audio files')
    parser.add_argument('audio_representation', type=str, help='Path to the audio representation config file')
    parser.add_argument('--annotations', default=None, type=str, help='Path to the annotations .csv')
    parser.add_argument('--annotation_step', default=0, type=float, help='Produce multiple time shifted representations views for each annotated  section by shifting the annotation  \
                window in steps of length step (in seconds) both forward and backward in time. The default value is 0.')
    parser.add_argument('--step_min_overlap', default=0.5, type=float, help='Minimum required overlap between the annotated section and the representation view, expressed as a fraction of whichever of the two is shorter. Only used if step > 0.')
    parser.add_argument('--labels', default=None, nargs='*', action=ParseKwargs, help='Specify a label mapping. Example: --labels background=0 upcall=1 will map labels with the string background to 0 and labels with string upcall to 1. \
        Any label not included in this mapping will be discarded. If None, will save every label in the annotation csv and will map the labels to 0, 1, 2, 3....')
    parser.add_argument('--table_name', default=None, type=str, help="Table name within the database where the data will be stored. Must start with a foward slash. For instance '/train'")
    parser.add_argument('--random_selections', default=None, nargs='+', type=int, help='Will generate random x number of samples with label y. --random_selections x y')
    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    parser.add_argument('--output', default=None, type=str, help='HDF5 dabase name. For isntance: db.h5')
    parser.add_argument('--overwrite', default=False, type=boolean_string, help='Overwrite the database. Otherwise append to it')
    args = parser.parse_args()

    create_db(**vars(args))

if __name__ == "__main__":
    main()