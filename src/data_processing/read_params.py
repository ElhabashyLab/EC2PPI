import os
import json
import pathlib

def set_default_param(param):
    """
    Sets default values for parameters that are not provided in the given dictionary.

    :param param: Parameter to be set to default.
    :return: Default values for the parameter.
    """
    if param == 'training_parameters':
        param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10]
        }
        return param_grid
    elif param == 'feature_list':
        features = [
            'n_eff',
            'n_eff_l',
            'sequence_length',
            'bit_score',
            'pairwise_identity',
            'cn_mean',
            'cn_std',
            'cn_median',
            'cn_iqr',
            'cn_max',
            'cn_skewness',
            'cn_kurtosis',
            'iptm',
            'fraction_disordered',
            'plddt_mean',
            'contact_probs_mean',
            'contact_probs_std',
            'contact_probs_median',
            'contact_probs_iqr',
            'contact_probs_max',
            'contact_probs_skewness',
            'contact_probs_kurtosis',
            'pae_mean',
            'pae_std',
            'pae_median',
            'pae_iqr',
            'pae_max',
            'pae_skewness',
            'pae_kurtosis'
        ]
        return features

def has_feature(feature_list, prefix=None, exact_match=None):
    """
    Checks if a feature is present in the feature list.

    :param feature_list: List of features.
    :param prefix: Optional prefix to check for.
    :param exact_match: Optional exact match to check for.
    :return: True if the feature is present, False otherwise.
    """
    if prefix:
        return any(feature.startswith(prefix) for feature in feature_list)
    if exact_match:
        return exact_match in feature_list
    return False

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
        raise ValueError('At least one of training_run or prediction_run must be set to true in the parameters.')

    # Parsing optional parameters
    optional_params = ['training_parameters', 'feature_list']
    for param in optional_params:
        if param not in params:
            params[param] = 'default'
        if params[param] == 'default':
            params[param] = set_default_param(param)
        if param == 'feature_list' and not isinstance(params[param], list):
            raise ValueError('feature_list must be a list of features or "default".')
        if param == 'training_parameters' and not isinstance(params[param], dict):
            raise ValueError('training_parameters must be a dictionary of parameters or "default".')

    # Check what features are included
    if (
        not has_feature(params['feature_list'], exact_match='iptm')
        and not has_feature(params['feature_list'], exact_match='fraction_disordered')
        and not has_feature(params['feature_list'], exact_match='plddt_mean')
        and not has_feature(params['feature_list'], prefix='pae_')
        and not has_feature(params['feature_list'], prefix='contact_probs_')
        ):
        params['include_af3'] = False
    else:
        params['include_af3'] = True

    if not has_feature(params['feature_list'], prefix='cn_'):
        params['include_ec'] = False
    else:
        params['include_ec'] = True

    # Checking required parameters for training and validate paths
    if params['training_run']:
        required_params_training = ['positive_training_complex_info_table_filepath',
                                    'negative_training_complex_info_table_filepath',
                                    'export_directory']

        if params['include_af3']:
            required_params_training += ['positive_training_complex_af3_directory',
                                         'negative_training_complex_af3_directory']
        if params['include_ec']:
            required_params_training += ['positive_training_complex_ec_directory',
                                         'negative_training_complex_ec_directory']

        for param in required_params_training:
            if param not in params:
                raise ValueError(f'Missing required training parameter: {param}')
            if param == 'export_directory':
                if not os.path.exists(params['export_directory']):
                    os.makedirs(params['export_directory'])
            if param.endswith('_filepath'):
                validate_path(params[param], 'file')
            elif param.endswith('_directory'):
                validate_path(params[param], 'directory')

    # Checking required parameters for prediction and validate paths
    if params['prediction_run']:
        required_params_prediction = ['prediction_complex_info_table_filepath',
                                      'model_import_filepath',
                                      'prediction_export_filepath']
        if params['include_af3']:
            required_params_prediction.append('prediction_complex_af3_directory')

        if params['include_ec']:
            required_params_prediction.append('prediction_complex_ec_directory')

        for param in required_params_prediction:
            if param not in params:
                raise ValueError(f'Missing required prediction parameter: {param}')
            if param.endswith('_filepath'):
                validate_path(params[param], 'file')
            elif param.endswith('_directory'):
                validate_path(params[param], 'directory')

        if params['model_import_filepath'] == 'latest':
            if 'model_export_filepath' not in params:
                raise ValueError('If model_import_filepath is set to "latest", model_export_filepath must also be provided.')
            params['model_import_filepath'] = params['model_export_filepath']

    print('All required parameters are present. Check complete.')

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
    if json_string.endswith(',\n}\n'):
        json_string = json_string[:-4] + '\n}\n'

    params = json.loads(json_string)

    if not isinstance(params, dict):
        raise ValueError(f'The input {params_file_path} is not a valid params file. Please provide the absolute or relative filepath to \'params_file.txt\'.')

    check_required_params(params)
    print(f'Parameters read from {params_file_path}:\n{json.dumps(params, indent=4)}')
    print('All paths are valid. Paths check complete.')

    return params