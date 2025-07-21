import os
import json

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

    print('All required parameters are present. Check complete.')

def read_params(params_file_path):
    """
    Reads the parameters from a given file path.

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

    return params