def command_run_model(args):
    from narw_detector.run_model import run_model
    del args['func']
    run_model(**args)

def command_test_model(args):
    from narw_detector.test_model import test_model
    del args['func']
    test_model(**args)

def command_train_model(args):
    from narw_detector.train_model import train_model
    del args['func']
    train_model(**args)

def command_create_db(args):
    from narw_detector.create_hdf5_db import create_db
    del args['func']
    create_db(**args)

def command_adapt_model(args):
    from narw_detector.adapt_model import adapt_model
    del args['func']
    adapt_model(**args)


def main():
    import argparse

    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split('=')
                if value.isdigit():
                    value = int(value)
                getattr(namespace, self.dest)[key] = value

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser = argparse.ArgumentParser("Bio North Atlantic right whale Detector")
    subparsers = parser.add_subparsers(help='sub-command help')

    runner_parser = subparsers.add_parser("run-model", help='Model runner commands')
    runner_parser.add_argument('model', type=str, help='Saved model (*.kt)')
    runner_parser.add_argument('audio_folder', type=str, help='Folder with audio files')
    runner_parser.add_argument('--file_list', default=None, type=str, help='A csv or .txt file where each row (or line) is the name of a file to detect within the audio folder. By default, all files will be processed.')
    runner_parser.add_argument('--threshold', default=0.5, type=float, help='Detection threshold (must be between 0 and 1)')
    runner_parser.add_argument('--step_size', default=None, type=float, help='Step size in seconds. If not specified, the '\
        'step size is set equal to the duration of the audio representation.')
    runner_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    runner_parser.add_argument('--output', default=None, type=str, help='Detection output .csv. For instance: detections/my_detections.csv')
    runner_parser.add_argument('--save_format', default=None, type=str, help='To save in raven format specify --save_format raven')
    runner_parser.add_argument('--merge_detections', default=False, type=boolean_string, help='Merge adjacent or overlapping detections.')
    runner_parser.add_argument('--overwrite', default=True, type=boolean_string, help='Overwrites the detections, otherwise appends to it. Allows resuming processing files.')
    runner_parser.set_defaults(func=command_run_model)


    train_parser = subparsers.add_parser("train-model", help='Model train options')
    train_parser.add_argument('model_recipe', type=str, help='Model recipe file')
    train_parser.add_argument('hdf5_db', type=str, help='HDF5 Database')
    train_parser.add_argument('audio_representation', type=str, help='Audio repsentation config file. It will be saved as metadata with the model.')
    train_parser.add_argument('--train_table', default='/', type=str, help="The table within the hdf5 database where the training data is stored. For example, /train")
    train_parser.add_argument('--val_table', default=None, type=str, help="The table within the hdf5 database where the validation data is stored. For example, /val. Optional.")
    train_parser.add_argument('--epochs', default=20, type=int, help='The number of epochs')
    train_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    train_parser.add_argument('--output_dir', default=None, type=str, help='Output directory')
    train_parser.add_argument('--model_output', default=None, type=str, help='Name of the model output file')
    train_parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    train_parser.set_defaults(func=command_train_model)


    adapt_parser = subparsers.add_parser("adapt-model", help="Adapt model options")
    adapt_parser.add_argument('model', type=str, help='Pre-trained model')
    adapt_parser.add_argument('hdf5_db', type=str, help='HDF5 Database')
    adapt_parser.add_argument('--train_table', default='/', type=str, help="The table within the hdf5 database where the training data is stored. For example, /train")
    adapt_parser.add_argument('--val_table', default=None, type=str, help="The table within the hdf5 database where the validation data is stored. For example, /val. Optional.")
    adapt_parser.add_argument('--epochs', default=20, type=int, help='The number of epochs')
    adapt_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    adapt_parser.add_argument('--audio_representation', type=str, help='Audio repsentation config file to transform the data. If not given, will use the one saved with the model.')
    adapt_parser.add_argument('--output_dir', default=None, type=str, help='Output directory. Will save the model output and associated files.')
    adapt_parser.add_argument('--model_output', default=None, type=str, help='Name of the model output file')
    adapt_parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    adapt_parser.set_defaults(func=command_adapt_model)

    test_parser = subparsers.add_parser("test-model", help='Model test options')
    test_parser.add_argument('model', type=str, help='Saved model (*.kt)')
    test_parser.add_argument('hdf5_db', type=str, help="hdf5 database path")
    test_parser.add_argument('--table_name', default=None, type=str, help="Table name within the database where the test data is stored. Must start with a foward slash. For instance '/test'. If not given, the root '/' path will be used")
    test_parser.add_argument('--threshold', default=0.5, type=float, help='Detection threshold (must be between 0 and 1)')
    test_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    test_parser.add_argument('--output_dir', default=None, type=str, help='The folder where the results will be saved.')
    test_parser.set_defaults(func=command_test_model)

    db_parser = subparsers.add_parser("create-db", help='Create hdf5 database options')
    db_parser.add_argument('data_dir', type=str, help='Path to the directory containing the audio files')
    db_parser.add_argument('audio_representation', type=str, help='Path to the audio representation config file')
    db_parser.add_argument('--annotations', default=None, type=str, help='Path to the annotations .csv')
    db_parser.add_argument('--annotation_step', default=0, type=float, help='Produce multiple representations view for each annotated  section by shifting the annotation  \
                window in steps of length step (in seconds) both forward and backward in time. The default value is 0.')
    db_parser.add_argument('--step_min_overlap', default=0.5, type=float, help='Minimum required overlap between the annotated section and the representation view, expressed as a fraction of whichever of the two is shorter. Only used if step > 0.')
    db_parser.add_argument('--labels', default=None, nargs='*', action=ParseKwargs, help='Specify a label mapping. Example: --labels background=0 upcall=1 will map labels with the string background to 0 and labels with string upcall to 1. \
        Any label not included in this mapping will be discarded. If None, will save every label in the annotation csv and will map the labels to 0, 1, 2, 3....')
    db_parser.add_argument('--table_name', default=None, type=str, help="Table name within the database where the data will be stored. Must start with a foward slash. For instance '/train'")
    db_parser.add_argument('--random_selections', default=None, nargs='+', type=int, help='Will generate random x number of samples with label y. --random_selections x y')
    db_parser.add_argument('--seed', default=None, type=int, help='Seed for random number generator')
    db_parser.add_argument('--output', default=None, type=str, help='HDF5 dabase name. For isntance: db.h5')
    db_parser.add_argument('--overwrite', default=False, type=boolean_string, help='Overwrite the database. Otherwise append to it')
    db_parser.set_defaults(func=command_create_db)

    args = parser.parse_args()
    args.func(vars(args))

if __name__ == "__main__":
    main()
    
