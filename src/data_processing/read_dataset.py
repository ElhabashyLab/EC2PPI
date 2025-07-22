import pathlib
import pandas as pd
from src.utils.protein import Protein
from src.utils.protein_pair import ProteinPair

def read_training_dataset(params):
    """
    Reads the training dataset based on the provided parameters.

    :param params: Dictionary containing parameters for reading the dataset.
    :return: List of PPI objects representing the training dataset.
    """

    training_complex_info_table_filepath = params['training_complex_info_table_filepath']
    training_complex_directory = params['training_complex_directory']
    training_complex_ec_directory = params['training_complex_ec_directory']
    training_complex_af3_directory = params['training_complex_af3_directory']

    protein_pairs = read_dataset(training_complex_info_table_filepath, training_complex_directory, training_complex_ec_directory, training_complex_af3_directory)
    return protein_pairs

def read_applied_dataset(params):
    """
    Reads the use case dataset based on the provided parameters.

    :param params: Dictionary containing parameters for reading the dataset.
    :return: List of PPI objects representing the use case dataset.
    """
    pass

def read_dataset(info_table_filepath, complex_directory, complex_ec_directory, complex_af3_directory):
    """
    Reads the dataset from the specified file paths.

    :param info_table_filepath:
    :param complex_directory:
    :param complex_ec_directory:
    :param complex_af3_directory:
    :return:
    """
    # Validate the file paths
    info_table_filepath = validate_path(info_table_filepath, 'file')
    #complex_directory = validate_path(complex_directory, 'directory')
    #complex_ec_directory = validate_path(complex_ec_directory, 'directory')
    #complex_af3_directory = validate_path(complex_af3_directory, 'directory')

    protein_pairs = []
    # Read the info table
    protein_pairs = read_info_table(info_table_filepath)

    return protein_pairs

def validate_path(path, expected_type):
    """
    Validates the file paths in the parameters dictionary.

    :param path: Path to be validated.
    :param expected_type: Type of path ('file' or 'directory').
    :raises ValueError: If any path is invalid.
    """
    path = pathlib.Path(path)#.resolve()

    if expected_type == 'file' and not path.is_file():
        raise ValueError(f'The input {path} is not a valid filepath. Please provide the absolute or relative filepath to \'params_file.txt\'.')
    elif expected_type == 'directory' and not path.is_dir():
        raise ValueError(f'The input {path} is not a valid directory. Please provide the absolute or relative filepath to \'params_file.txt\'.')

    return path

def read_info_table(info_table_filepath):
    """
    Reads the info table from the specified file path.

    :param info_table_filepath: Path to the info table file.
    :return: DataFrame containing the info table.
    """
    pps = []
    try:
        df = pd.read_csv(info_table_filepath, sep=',')
        for index, row in df.iterrows():
            protein1 = Protein(uniprot_id=row['uid1'])
            protein2 = Protein(uniprot_id=row['uid2'])
            pps.append(ProteinPair(protein1=protein1, protein2=protein2))
        return pps
    except Exception as e:
        raise ValueError(f'Error reading info table from {info_table_filepath}: {e}') from e