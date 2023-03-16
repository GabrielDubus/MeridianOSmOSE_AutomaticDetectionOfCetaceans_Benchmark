""" bio NARW_detector - Adapting a Classifier

    This module of CLI (command line interface) adapts a pre-trained neural network model through transfer learning to a new dataset. 
    A use case of this tool would be to adapt a model that had been trained with data from one environment to detect NARW in another environment.
    The inputs are a pre-trained ketos model and new audio representations of fixed length stored in an hdf5 database.

    Example of use:

        $train_model.py model.kt narw.h5 --train_table '/train' --val_table '/val'

    where `model.kt` is a pretrained ketos model, `narw.h5` is a hdf5 database containing the new data, `/train` is the path to the training data within the hdf5 database and `/val` is the path to the validation data within the hdf5 database.

    If no training table is given, the root group '/' is used. If no validation table is given, model adaptation will proceed without a validation step.


    The result will be a ketos (.kt) model file with the transfer learning model.

    The tool will output a log .csv with detailed metrics for each training epoch. Each row of the .csv file has the following column:

    - epoch: epoch number
    - loss: the loss value computed at the end of the training epoch. Depends on the loss function defined in the configuration file
    - dataset: Whether the row belongs to the training or validation dataset
    - metric columns: Any metric defined in the configuration file. Each metric will be a column. Common metrics are Binary accuracy, Precision and Recall.

    For more information see the README.md file.
"""

import numpy as np
import os
import ketos as kt
from pathlib import Path
from narw_detector.utils.train_model_utils import which_nn_interface, import_nn_interface
from ketos.data_handling.data_feeding import BatchGenerator, JointBatchGen
import ketos.data_handling.database_interface as dbi
from ketos.neural_networks.dev_utils.nn_interface import RecipeCompat, NNInterface
import tensorflow as tf


def adapt_model(model, hdf5_db, train_table=None, val_table=None, batch_size=32, epochs=5, audio_representation=None, output_dir=None, model_output=None, seed=None):
    if seed is not None:
        np.random.seed(seed) #set random seeds
        tf.random.set_seed(seed)

    nn_recipe, model_audio_representation, nn_module = which_nn_interface(model)
    nn_interface = import_nn_interface(name=nn_recipe['interface'], module_path=nn_module)

    nn = nn_interface.load(model)

    if audio_representation is None: # an audio repreesntation was not given so we will use the one saved with the model
        audio_representation = model_audio_representation
    
    # Open the database
    db = dbi.open_file(hdf5_db, 'r')
    
    if train_table is None:
        train_table = "/"

    # create batch generators
    # This code will create a generator for each subgroup within a table. This will ensure that each batch will contain the same number of samples for each class
    # For intance if the table has /train/positives and /train/negatives, the code will create 2 generators, with batch size = batch_size/2 for each label and join them later
    train_gens = []

    # Getting Batch size by looping through the hdf5 file and searching for leaf nodes which are an object with name Table
    # I do not believe there is a more efficient way of gettign the number of leaf nodes
    train_batch_size = int(batch_size / sum(1 for _ in db.walk_nodes(train_table, "Table")))
    for group in db.walk_nodes(train_table, "Table"): # Here we are iterating over nodes with classname=Table these are leaf nodes that actually contain data
        train_generator = BatchGenerator(batch_size=train_batch_size,
                            data_table=group, map_labels=False,
                            output_transform_func=nn_interface.transform_batch,
                            shuffle=True, refresh_on_epoch_end=True, x_field="data")
        train_gens.append(train_generator)

    # join generators
    train_generator = JointBatchGen(train_gens, n_batches="min", shuffle_batch=False, map_labels=False, reset_generators=False)

    
    if val_table is not None:
        val_gens = []
        val_batch_size = int(batch_size / sum(1 for _ in db.walk_nodes(val_table, "Table")))
        for group in db.walk_nodes(val_table, "Table"):
            val_generator = BatchGenerator(batch_size=val_batch_size,
                                    data_table=group, map_labels=False,
                                    output_transform_func=nn_interface.transform_batch,
                                    shuffle=True, refresh_on_epoch_end=False, x_field='data')
            val_gens.append(val_generator)

        # join generators
        val_generator = JointBatchGen(val_gens, n_batches="min", shuffle_batch=False, map_labels=False, reset_generators=False)

    # Replacing optimizer with the same one but a lower learning rate
    optimizer = nn_recipe["optimizer"]
    optimizer["parameters"]["learning_rate"] = optimizer["parameters"]["learning_rate"] / 10
    dec_opt = RecipeCompat(optimizer['recipe_name'], NNInterface.valid_optimizers[optimizer['recipe_name']], **optimizer["parameters"])
    nn.optimizer = dec_opt
    
    if output_dir is None:
        output_dir = './trained_models'

    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # output paths
    nn.log_dir = output_dir
    nn.checkpoint_dir = os.path.join(output_dir, 'checkpoints')

    # set generators
    nn.train_generator = train_generator
    if val_table is not None:
        nn.val_generator = val_generator

    if model_output is None:
        model_output = "adapted_model.kt"
    
    # Check if name already has extension otherwise add it
    root, ext = os.path.splitext(model_output)
    if not ext:
        ext = '.kt'
    model_path = os.path.join(output_dir, root + ext)

    #Train the network
    print("\n Training Starting ...")
    nn.train_loop(n_epochs=epochs, verbose=True, log_csv=True, csv_name="adapt_log.csv", validate=(val_table is not None), checkpoint_freq=epochs)

    # Here I am resetting the optimizer to the original learning rate so that the model is saved with the original one.
    # We do this is that otherwise, subsequent adaptations would have a exponentially lower and lower larning rate.
    # Maybe there is a better way of doing this, but for now, this sufices.
    optimizer["parameters"]["learning_rate"] = optimizer["parameters"]["learning_rate"] * 10
    dec_opt = RecipeCompat(optimizer['recipe_name'], NNInterface.valid_optimizers[optimizer['recipe_name']], **optimizer["parameters"])
    nn.optimizer = dec_opt
    
    metadata = nn_recipe
    metadata['ketos_version'] = kt.__version__
    if nn_module is not None: 
        metadata['nn_module'] = nn_module

    nn.save(output_name=model_path, audio_repr=audio_representation, metadata=metadata, extra=nn_module)

    db.close()

def main():
    import argparse

    # parse command-line args
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, help='Pre-Trained model')
    parser.add_argument('hdf5_db', type=str, help='HDF5 Database file path')
    parser.add_argument('--train_table', default='/', type=str, help="The table within the hdf5 database where the training data is stored. For example, /train")
    parser.add_argument('--val_table', default=None, type=str, help="The table within the hdf5 database where the validation data is stored. For example, /val. Optional.")
    parser.add_argument('--epochs', default=20, type=int, help='The number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--audio_representation', type=str, help='Audio repsentation config file to transform the data. If not given, will use the one saved with the model.')
    parser.add_argument('--output_dir', default=None, type=str, help='Output directory. Will save the model output and associated files.')
    parser.add_argument('--model_output', default=None, type=str, help='Name of the model output file')
    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    args = parser.parse_args()

    adapt_model(**vars(args))

if __name__ == "__main__":
    main()