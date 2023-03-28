# Task 2-a - Detecting North Atlantic right whales in different envirionments
## Description

The goal of this task is to produce a model that is able to detect North Atlantic right whale (NARW) upcalls accross multiple soundscapes with a reasonable performance. The training dataset is a collection of recordings obtained in the Gulf of Maine and the test dataset are recordings collected in the Emerald Basin. The focus of this task is on the data-preparation and training startegies used for producing robust models on different underwater environments.


The models will be evaluated on how well they generalize to unseen data from other regions. The goal will be to maximize precision / recall and minimize the FPR per hour or recording.


Annotations for each soundscape will contain at least the following columns:
| filename | start | end |
| ------- | ------ | ---- |

where filename is the name of the file that contains a NARW upcall, start is the start time in seconds where the annotation begins and end is the end time in seconds where the annotation ends.



# Running the sample code 

## Requirements

Python > 3.9.10

It is recommended to create a new python environment to run the code and install the dependencies.


Install the dependencies with:

```
pip install -r requirements.txt
```

To run the code simply issue the commands in your CLI. Ensure you are in the correct working directory.

## Creating the database

* `config_files/spec_config.json` contains the spectrogram configuration we will be using
* `config_files/resnet_recipe.json` contains the NN architecture we will be using

First we add the NARW annotations (positives) to the database:

```
python -m narw_detector.create_hdf5_db data/train/narw/ config_files/spec_config.json --annotations data/train/annotations_train.csv --labels 1=1 --table_name /train --output db.h5
```

Next we add the background data (negatives):

```
python -m narw_detector.create_hdf5_db data/train/background/ config_files/spec_config.json --random_selections 53 0 --table_name /train --output db.h5
```


<!-- Now, lets create the validation table: -->

<!-- ```
python -m narw_detector.create_hdf5_db data/val/narw/ config_files/spec_config.json --annotations data/val/annotations_val.csv --labels 1=1 --table_name /val --output db.h5

python -m narw_detector.create_hdf5_db data/val/background/ config_files/spec_config.json --random_selections 20 0 --table_name /val --output db.h5
``` -->

## Train a Model

With the database created we can train a NN. Lets use the resnet archictecture defined in the configuration file.

```
python -m narw_detector.train_model config_files/resnet_recipe.json db.h5 config_files/spec_config.json --train_table /train --epochs 5 --output_dir trained_model/ --model_output narw_detector.kt
```

## Run the Model

To run the model on a directory with audio files:

```
python -m narw_detector.run_model trained_model/narw_detector.kt data/test/ --output detections/detections.csv
```

The detections will be saved on to a csv file in the `detections` folder.

## Evaluate on test data

Finally to test the detections:

```
python narw_detector/utils/detector_performance.py detections/detections.csv data/test/annotations_test.csv
```
