import pathlib as pathlib
import os
import pandas as pd
from src.utils.protein import Protein
from src.utils.protein_pair import ProteinPair

def read_training_dataset(params):
    """
    Reads the training dataset based on the provided parameters.

    :param params: Dictionary containing parameters for reading the dataset.
    :return: List of PPI objects representing the training dataset.
    """

    # Extract parameters for reading the positive dataset
    positive_training_complex_info_table_filepath = params['positive_training_complex_info_table_filepath']
    positive_training_complex_ec_directory = params['positive_training_complex_ec_directory']
    positive_training_complex_af3_directory = params['positive_training_complex_af3_directory']

    # Read the positive training dataset
    positive_protein_pairs = read_dataset(
        info_table_filepath=positive_training_complex_info_table_filepath,
        complex_ec_directory=positive_training_complex_ec_directory,
        complex_af3_directory=positive_training_complex_af3_directory,
        label=1
    )

    # Extract parameters for reading the negative dataset
    negative_training_complex_info_table_filepath = params['negative_training_complex_info_table_filepath']
    negative_training_complex_ec_directory = params['negative_training_complex_ec_directory']
    negative_training_complex_af3_directory = params['negative_training_complex_af3_directory']

    # Read the negative training dataset
    negative_protein_pairs = read_dataset(
        info_table_filepath=negative_training_complex_info_table_filepath,
        complex_ec_directory=negative_training_complex_ec_directory,
        complex_af3_directory=negative_training_complex_af3_directory,
        label=0
    )

    # Combine positive and negative protein pairs
    protein_pairs = positive_protein_pairs + negative_protein_pairs

    return protein_pairs

def read_applied_dataset(params):
    """
    Reads the use case dataset based on the provided parameters.

    :param params: Dictionary containing parameters for reading the dataset.
    :return: List of PPI objects representing the use case dataset.
    """
    pass

def get_filepath_from_prefix(directory, prefix):
    """
    Returns the file path for a given prefix in the specified directory.

    :param directory: Directory to search for the file.
    :param prefix: Prefix of the file to find.
    :return: File path as a string.
    """

    all_files = [entry.path for entry in os.scandir(directory) if entry.is_file()]
    for filepath in all_files:
        if os.path.basename(filepath).startswith(prefix):
            return filepath

    raise FileNotFoundError(f'No file found with prefix {prefix} in directory {directory}')

def read_dataset(info_table_filepath, complex_ec_directory, complex_af3_directory, label):
    """
    Reads the dataset from the specified file paths.

    :param info_table_filepath:
    :param complex_ec_directory:
    :param complex_af3_directory:
    :param label: Label for the protein pairs (1 for positive, 0 for negative).
    :return: List of ProteinPair objects representing the dataset.
    """

    # Read uniprot ids for each pair of proteins from the info table
    ec_directory = pathlib.Path(complex_ec_directory)
    af3_directory = pathlib.Path(complex_af3_directory)
    protein_pairs = []
    try:
        df = pd.read_csv(info_table_filepath, sep=',')
        for index, row in df.iterrows():
            protein1 = Protein(uniprot_id=row['uid1'],n_eff=row['Neff1'], n_eff_l=row['NeffL1'], sequence_length=row['seq1_len'], bit_score=row['bit1'])
            protein2 = Protein(uniprot_id=row['uid2'],n_eff=row['Neff2'], n_eff_l=row['NeffL2'], sequence_length=row['seq2_len'], bit_score=row['bit2'])
            prefix = row['prefix']
            ec_filepath = get_filepath_from_prefix(ec_directory, prefix)
            #af3_filepath = get_filepath_from_prefix(af3_directory, prefix)
            protein_pairs.append(ProteinPair(prefix=prefix, protein1=protein1, protein2=protein2, ec_filepath=ec_filepath, label=label))
    except Exception as e:
        raise ValueError(f'Error reading data: {e}') from e

    return protein_pairs
