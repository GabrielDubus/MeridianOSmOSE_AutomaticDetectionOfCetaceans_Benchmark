import os
import json
import yaml
from shutil import rmtree
from ketos.data_handling.parsing import load_audio_representation
import sys
import importlib
from zipfile import ZipFile


def load_config(path, model_name):
    ''' Load the relevant sections of the model training configuration yaml file.

        Args:
            path: str
                Full path to the configuration yaml file.
            model_name: str
                Model for which configuration should be loaded

        Returns:
            parsed: dict
                Parsed contents of configuration file.
            raw: dict
                Unparsed contents
            audio_repr: dict
                Audio representations
    '''
    with open(path, "r") as stream:
        d = yaml.safe_load(stream)

        raw, audio_repr = parse_config(d, model_name, os.path.dirname(path))

        return raw, audio_repr


def parse_config(d, model_name, rel_dir=None):
    ''' Parse the model training configuration.

        Args:
            d: dict
                Dictionary containing the database configuration
            model_name: str
                Model for which configuration should be loaded
            rel_dir: str
                If specified, relative paths will be resolved relative 
                to this directory location.
                
        Returns:
            parsed: dict
                Parsed configuration
            raw dict
                Unparsed configuration
            audio_repr: dict
                Audio representations 
    '''
    # load the audio representations
    path = resolve_path(d["representation_path"], rel_dir)
    audio_repr = load_audio_representation(path)

    assert model_name in d["models"].keys(),\
        f"Model training configuration could not be found for {model_name}"

    # parse model train config    
    raw = d["models"][model_name]

    # get names of audio representations that are being used
    ar_names = []
    if not isinstance(raw["representation"], list):
        raw["representation"] = [raw["representation"]]

    for val in raw["representation"]:
        if isinstance(val, str): ar_names.append(val)
        else: ar_names += val

    # only keep the audio representations that are being used
    audio_repr = {key: val for key,val in audio_repr.items() if key in ar_names}

    return raw, audio_repr

ketos_nn_modules = {
    "DenseNetInterface": "ketos.neural_networks.densenet",
    "ResNetInterface": "ketos.neural_networks.resnet",
    "ResNet1DInterface": "ketos.neural_networks.resnet",
    "CNNInterface": "ketos.neural_networks.cnn",
    "InceptionInterface": "ketos.neural_networks.inception"
}


def import_nn_interface(name, module_path=None):
    ''' Import neural network interface class

        Note: when importing a non-standard ketos interface from a zip archive, 
        a temporary folder named `.kt-tmp` is created for storing the Python module.

        Args:
            name: str
                Name of the interface class, e.g., 'DenseNetInterface'
            module_path: str or tuple
                Path to the Python module file, e.g., 'nn/densenet.py'. Only required if the 
                interface is not a standard ketos interface. 
                Can also be a tuple consisting of 1) the path to a zip archive, 2) the path 
                of the Python module within the zip archive
            
        Returns:
            : class derived from ketos.neural_networks.nn_interface.NNInterface
    '''
    if name in ketos_nn_modules.keys():
        module_name = ketos_nn_modules[name]
    
    else: #attempt to import user-specified module
        assert module_path is not None, "Path to Python module is required for loading user-specified "\
            f"neural network interface {name}"

        if isinstance(module_path, tuple): #if module is within zip archive
            # create new temporary folder
            tmp_folder = ".kt-tmp/"
            if os.path.isdir(tmp_folder):
                rmtree(tmp_folder)

            # extract module to the temporary folder
            with ZipFile(module_path[0], 'r') as zip:
                zip.extract(member=module_path[1], path=tmp_folder)

            # point to the extracted py file
            module_path = os.path.join(tmp_folder, module_path[1])

        assert os.path.exists(module_path), f"Could not find module {module_path}"

        sys.path.append(os.path.dirname(os.path.abspath(module_path)))
        module_name = os.path.basename(module_path)[:-3] #strip *.py extension

    return getattr(importlib.import_module(module_name), name)

def resolve_path(path, rel_dir=None):
    ''' Resolve a relative path.

        If an absolute path is provided, the return value
        as the input value.

        Args: 
            path: str
                Relative path.
            rel_dir: str
                Top directory for relative paths

        Returns:
            path: str
                The absolute path obtained by joining the top 
                directory and the relative path.
    '''
    if not os.path.isabs(path) and rel_dir is not None:
        path = os.path.join(rel_dir, path)                    

    return path

def _extract_json_from_archive(archive, path):
    ''' Extract contents of json file from within zip archive file.

        Args:
            archive: str
                Full path to zip archive file
            path: str
                Path to json file within zip archive

        Returns:
            res: dict
                Contents of json file
    '''
    tmp_folder = ".tmp/"

    # create new temporary folder
    if os.path.isdir(tmp_folder):
        rmtree(tmp_folder)

    # extract contents of zip archive to temporary folder
    with ZipFile(archive, 'r') as zip:
        zip.extractall(path=tmp_folder)

    if not os.path.exists(os.path.join(tmp_folder, path)):
        rmtree(tmp_folder)
        return None

    # load contents of json file
    with open(os.path.join(tmp_folder, path), 'r') as json_file:
        res = json.load(json_file)

    # clean up
    rmtree(tmp_folder)

    return res

def which_nn_interface(path):
    ''' Extract information about neural network interface from saved model file (*.kt)

        Args:
            path: str
                Path to saved model

        Returns:
            name: str
                The name of the neural network interface
            module_path: str or tuple
                Only relevant if the interface is not a standard ketos interface.
                The path to the Python module, or a tuple consisting of 1) the path to 
                a zip archive, 2) the path of the Python module within the zip archive.
    '''
    recipe = _extract_json_from_archive(path, "recipe.json")
    metadata = _extract_json_from_archive(path, "metadata.json")
    audio_representation = _extract_json_from_archive(path, "audio_repr.json")

    module_path = None
    if metadata is not None:
        module_path = metadata.get('nn_module')
        if module_path is not None:
            module_name = os.path.basename(module_path)
            with ZipFile(path, 'r') as zip:
                if module_name in zip.namelist():
                    module_path = (path, module_name)

    return recipe, audio_representation, module_path