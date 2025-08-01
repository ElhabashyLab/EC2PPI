import os
import json
import pathlib

def check_required_params(params):
    """
    Checks if all required parameters are present in the given dictionary.

    :param params: Dictionary containing parameters.
    :raises ValueError: If any required parameter is missing.
    """

    print('Checking required parameters...')
    required_params = ['training_run', 'prediction_run']

    for param in required_params:
        if param not in params:
            raise ValueError(f'Missing required parameter: {param}')
    if not (params['training_run'] or params['prediction_run']):
        raise ValueError('At least one of training_run or prediction_run must be set to True in the parameters.')

    if params['training_run']:
        required_params_training = ['training_complex_info_table_filepath',
                                    'training_complex_directory',
                                    'training_complex_ec_directory',
                                    'training_complex_af3_directory',
                                    'model_export_filepath',]
        for param in required_params_training:
            if param not in params:
                raise ValueError(f'Missing required training parameter: {param}')

    if params['prediction_run']:
        required_params_prediction = ['prediction_complex_info_table_filepath',
                                      'prediction_complex_directory',
                                      'prediction_complex_ec_directory',
                                      'prediction_complex_af3_directory',
                                      'model_import_filepath',
                                      'prediction_export_filepath']
        for param in required_params_prediction:
            if param not in params:
                raise ValueError(f'Missing required prediction parameter: {param}')
        if params['model_import_filepath'] == 'latest':
            if 'model_export_filepath' not in params:
                raise ValueError('If model_import_filepath is set to "latest", model_export_filepath must also be provided.')
            params['model_import_filepath'] = params['model_export_filepath']

    print('All required parameters are present. Check complete.')


def check_paths(params):
    """ Checks if the file paths in the parameters dictionary are valid.
    :param params: Dictionary containing parameters.
    :raises ValueError: If any path is invalid.
    """

    for key, value in params.items():
        if key.endswith('_filepath'):
            validate_path(value, 'file')
        elif key.endswith('_directory'):
            validate_path(value, 'directory')

def validate_path(path, expected_type):
    """
    Validates the file paths in the parameters dictionary.

    :param path: Path to be validated.
    :param expected_type: Type of path ('file' or 'directory').
    :raises ValueError: If any path is invalid.
    """
    path = pathlib.Path(path)#.resolve()

    if expected_type == 'file' and not path.is_file():
        #raise ValueError(f'The input {path} is not a valid filepath. Please provide the absolute or relative filepath to \'params_file.txt\'.')
        print(f'WARNING: The input {path} is not a valid filepath. Please provide the absolute or relative filepath to \'params_file.txt\'.')
    elif expected_type == 'directory' and not path.is_dir():
        #raise ValueError(f'The input {path} is not a valid directory. Please provide the absolute or relative filepath to \'params_file.txt\'.')
        print(f'WARNING: The input {path} is not a valid directory. Please provide the absolute or relative filepath to \'params_file.txt\'.')

    return path

def read_params(params_file_path):
    """
    Reads the parameters from a given file path and checks for required parameters.

    :param params_file_path: Path to the parameters file.
    :return: Dictionary containing the parameters.
    """

    if not os.path.isfile(params_file_path):
        raise FileNotFoundError(f'The input {params_file_path} is not a valid filepath. Please provide the absolute or relative filepath to \'params_file.txt\'.')

    with open(params_file_path) as f:
        data = f.readlines()

    json_string = ''
    for d in data:
        d = d.strip()
        if not d.startswith('#') and len(d) > 0:
            json_string += d + '\n'

    params = json.loads(json_string)

    if not isinstance(params, dict):
        raise ValueError(f'The input {params_file_path} is not a valid params file. Please provide the absolute or relative filepath to \'params_file.txt\'.')

    check_required_params(params)
    print(f'Parameters read from {params_file_path}:\n{json.dumps(params, indent=4)}')
    check_paths(params)
    print('All paths are valid. Paths check complete.')

    return params