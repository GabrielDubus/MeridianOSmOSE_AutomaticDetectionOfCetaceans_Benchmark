import numpy as np
import os
import yaml
import ketos as kt
from pathlib import Path
from narw_detector.utils.train_model_utils import import_nn_interface
from ketos.data_handling.data_feeding import BatchGenerator, JointBatchGen
import ketos.data_handling.database_interface as dbi
import tensorflow as tf


def train_model(model_recipe, hdf5_db, audio_representation, train_table=None, val_table=None, batch_size=32, epochs=20, output_dir=None, model_output=None, seed=None):
    if seed is not None:
        np.random.seed(seed) #set random seeds
        tf.random.set_seed(seed)

    # Import the neural-network interface class
    with open(model_recipe, 'r') as f:
        nn_recipe = yaml.safe_load(f)

    nn_module = nn_recipe.get('nn_module')
    nn_interface = import_nn_interface(name=nn_recipe['interface'], module_path=nn_module)

    # Open the database
    db = dbi.open_file(hdf5_db, 'r')

    if train_table is None:
        train_table = "/"

    # create batch generators
    # This code will create a generator for each subgroup within a table. This will ensure that each batch will contain the same number of samples for each class
    # For intance if the table has /train/positives and /train/negatives, the code will create 2 generators, with batch size = batch_size/2 for each label and join them later
    train_gens = []
    train_batch_size = int(batch_size / sum(1 for _ in db.walk_nodes(train_table, "Table")))
    for group in db.walk_nodes(train_table, "Table"):
        generator = BatchGenerator(batch_size=train_batch_size,
                            data_table=group, map_labels=False,
                            output_transform_func=nn_interface.transform_batch,
                            shuffle=True, refresh_on_epoch_end=True, x_field="data")
        train_gens.append(generator)
        

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

    # build network
    nn = nn_interface.build(model_recipe)

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
        model_output = "trained_model.kt"

    # Check if name already has extension otherwise add it
    root, ext = os.path.splitext(model_output)
    if not ext:
        ext = '.kt'
    model_path = os.path.join(output_dir, root + ext)

    # train the network
    print("\n Training Starting ...")
    nn.train_loop(n_epochs=epochs, verbose=True, log_csv=True, validate=(val_table is not None), checkpoint_freq=epochs)

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
    
    parser.add_argument('model_recipe', type=str, help='Model recipe file')
    parser.add_argument('hdf5_db', type=str, help='HDF5 Database file path')
    parser.add_argument('audio_representation', type=str, help='Audio repsentation config file. It will be saved as metadata with the model.')
    parser.add_argument('--train_table', default='/', type=str, help="The table within the hdf5 database where the training data is stored. For example, /train")
    parser.add_argument('--val_table', default=None, type=str, help="The table within the hdf5 database where the validation data is stored. For example, /val. Optional.")
    parser.add_argument('--epochs', default=20, type=int, help='The number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--output_dir', default=None, type=str, help='Output directory')
    parser.add_argument('--model_output', default=None, type=str, help='Name of the model output file')
    parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    args = parser.parse_args()

    train_model(**vars(args))

if __name__ == "__main__":
    main()