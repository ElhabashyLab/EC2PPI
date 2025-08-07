import pandas as pd
from src.utils.protein import Protein
from src.utils.protein_pair import ProteinPair
from src.utils.print import progress_bar


def features_from_ec_file(ec_filepath: str):
    """Extract features from the EC file.

    :param ec_filepath: Path to the EC file.
    :return: List of features extracted from the EC file.
    """
    df = pd.read_csv(ec_filepath)
    cn_values = df['cn']
    features = [
        cn_values.mean(),
        cn_values.std(),
        cn_values.median(),
        cn_values.quantile(0.75) - cn_values.quantile(0.25),
        cn_values.max(),
        cn_values.skew(),
        cn_values.kurtosis()
    ]

    return features

def calculate_features(protein_pair: ProteinPair):
    """Calculate features for a given protein pair.

    :param protein_pair: ProteinPair object containing the proteins and their attributes.
    :return: List of features for the protein pair.
    """
    features = [
        protein_pair.protein1.n_eff,
        protein_pair.protein2.n_eff,
        protein_pair.protein1.n_eff_l,
        protein_pair.protein2.n_eff_l,
        protein_pair.protein1.sequence_length,
        protein_pair.protein2.sequence_length,
        protein_pair.protein1.bit_score,
        protein_pair.protein2.bit_score,
        protein_pair.pairwise_identity
    ]

    # calculate features from EC file
    features.extend(features_from_ec_file(protein_pair.ec_filepath))

    # placeholder for AF3 features
    # features.extend(features_from_af3_file(protein_pair.af3_filepath))



    return features

def calculate_all_features(protein_pairs):
    """Calculate features for all protein pairs in the list.

    :param protein_pairs: List of ProteinPair objects.
    :return: None, modifies the protein_pairs in place.
    """
    print('Calculating all features')
    for i, protein_pair in enumerate(protein_pairs):
        protein_pair.features = calculate_features(protein_pair)
        progress_bar(i, len(protein_pairs))
