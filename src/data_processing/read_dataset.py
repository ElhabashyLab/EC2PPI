import os
import pandas as pd
from src.utils.protein import Protein
from src.utils.protein_pair import ProteinPair
from src.utils.print import progress_bar

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
    print('Reading positive training dataset...')
    positive_protein_pairs = read_dataset(
        info_table_filepath=positive_training_complex_info_table_filepath,
        ec_directory=positive_training_complex_ec_directory,
        af3_directory=positive_training_complex_af3_directory,
        label=1
    )


    # Extract parameters for reading the negative dataset
    negative_training_complex_info_table_filepath = params['negative_training_complex_info_table_filepath']
    negative_training_complex_ec_directory = params['negative_training_complex_ec_directory']
    negative_training_complex_af3_directory = params['negative_training_complex_af3_directory']

    # Read the negative training dataset
    print('Reading negative training dataset...')
    negative_protein_pairs = read_dataset(
        info_table_filepath=negative_training_complex_info_table_filepath,
        ec_directory=negative_training_complex_ec_directory,
        af3_directory=negative_training_complex_af3_directory,
        label=0
    )

    # Combine positive and negative protein pairs
    if len(positive_protein_pairs) > len(negative_protein_pairs):
        print(f'Warning: More positive protein pairs ({len(positive_protein_pairs)}) than negative ({len(negative_protein_pairs)}).')
        positive_protein_pairs = positive_protein_pairs[:len(negative_protein_pairs)]
    elif len(negative_protein_pairs) > len(positive_protein_pairs):
        print(f'Warning: More negative protein pairs ({len(negative_protein_pairs)}) than positive ({len(positive_protein_pairs)}).')
        negative_protein_pairs = negative_protein_pairs[:len(positive_protein_pairs)]

    protein_pairs = positive_protein_pairs + negative_protein_pairs

    print(f'Read {len(protein_pairs)} protein pairs for training.')

    return protein_pairs

def read_applied_dataset(params):
    """
    Reads the use case dataset based on the provided parameters.

    :param params: Dictionary containing parameters for reading the dataset.
    :return: List of PPI objects representing the use case dataset.
    """
    pass

def get_path_from_prefix(directory, prefix):
    """
    Returns the file path for a given prefix in the specified directory.

    :param directory: Directory to search for the file.
    :param prefix: Prefix of the file to find.
    :return: File path as a string.
    """

    all_files = [entry.path for entry in os.scandir(directory) if (entry.is_file()) or (entry.is_dir())]
    for path in all_files:
        if os.path.basename(path).startswith(prefix):
            return path

    print(f'No file/directory found with prefix {prefix} in directory {directory}')
    return None

def read_dataset(info_table_filepath, ec_directory, af3_directory, label):
    """
    Reads the dataset from the specified file paths.

    :param info_table_filepath:
    :param ec_directory:
    :param af3_directory:
    :param label: Label for the protein pairs (1 for positive, 0 for negative).
    :return: List of ProteinPair objects representing the dataset.
    """

    # Read uniprot ids for each pair of proteins from the info table
    protein_pairs = []
    try:
        df = pd.read_csv(info_table_filepath, sep=',')
        for index, row in df.iterrows():
            protein1 = Protein(
                uniprot_id=row['uid1'],
                n_eff=row['Neff1'],
                n_eff_l=row['NeffL1'],
                sequence_length=row['seq1_len'],
                bit_score=row['bit1']
            )
            protein2 = Protein(
                uniprot_id=row['uid2'],
                n_eff=row['Neff2'],
                n_eff_l=row['NeffL2'],
                sequence_length=row['seq2_len'],
                bit_score=row['bit2']
            )
            prefix = row['prefix']

            ec_filepath = get_path_from_prefix(ec_directory, prefix)
            if ec_filepath is None:
                continue

            af3_directory_single = get_path_from_prefix(af3_directory, prefix)
            if af3_directory_single is None:
                continue

            protein_pair = ProteinPair(
                prefix=prefix,
                protein1=protein1,
                protein2=protein2,
                ec_filepath=ec_filepath,
                af3_directory=af3_directory_single,
                label=label,
                pairwise_identity=row['pairwise_identity'])
            protein_pairs.append(protein_pair)

            progress_bar(index, len(df))

    except Exception as e:
        raise ValueError(f'Error reading data: {e}') from e

    return protein_pairs
