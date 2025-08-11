import pandas as pd
import json
from src.utils.protein import Protein
from src.utils.protein_pair import ProteinPair
from src.utils.print import progress_bar


def features_from_ec_file(ec_filepath: str, feature_list: list):
    """Extract features from the EC file.

    :param ec_filepath: Path to the EC file.
    :param feature_list: List of features to extract.
    :return: List of features extracted from the EC file.
    """
    df = pd.read_csv(ec_filepath)
    cn_values = df['cn']

    features = []

    if 'cn_mean' in feature_list:
        features.append(cn_values.mean())

    if 'cn_std' in feature_list:
        features.append(cn_values.std())

    if 'cn_median' in feature_list:
        features.append(cn_values.median())

    if 'cn_iqr' in feature_list:
        features.append(cn_values.quantile(0.75) - cn_values.quantile(0.25))

    if 'cn_max' in feature_list:
        features.append(cn_values.max())

    if 'cn_skewness' in feature_list:
        features.append(cn_values.skew())

    if 'cn_kurtosis' in feature_list:
        features.append(cn_values.kurtosis())

    return features

def features_from_af3_file(af3_filepath: str, feature_list: list):
    """Extract features from the AF3 file.

    :param af3_filepath: Path to the AF3 file.
    :param feature_list: List of features to extract.
    :return: List of features extracted from the AF3 file.
    """


    pass

def calculate_features(protein_pair: ProteinPair, feature_list: list):
    """Calculate features for a given protein pair.

    :param protein_pair: ProteinPair object containing the proteins and their attributes.
    :param feature_list: List of features to calculate.
    :return: List of features for the protein pair.
    """
    features = []

    if 'n_eff' in feature_list:
        features.append(protein_pair.protein1.n_eff)
        features.append(protein_pair.protein2.n_eff)

    if 'n_eff_l' in feature_list:
        features.append(protein_pair.protein1.n_eff_l)
        features.append(protein_pair.protein2.n_eff_l)

    if 'sequence_length' in feature_list:
        features.append(protein_pair.protein1.sequence_length)
        features.append(protein_pair.protein2.sequence_length)

    if 'bit_score' in feature_list:
        features.append(protein_pair.protein1.bit_score)
        features.append(protein_pair.protein2.bit_score)

    if 'pairwise_identity' in feature_list:
        features.append(protein_pair.pairwise_identity)

    # calculate features from EC file
    features.extend(features_from_ec_file(protein_pair.ec_filepath, feature_list))

    # placeholder for AF3 features
    # features.extend(features_from_af3_file(protein_pair.af3_filepath))

    return features

def calculate_all_features(protein_pairs, params):
    """Calculate features for all protein pairs in the list.

    :param protein_pairs: List of ProteinPair objects.
    :param params: Dictionary containing parameters, including the feature list.
    :return: None, modifies the protein_pairs in place.
    """
    print('Calculating all features...')
    feature_list = params['feature_list']
    for i, protein_pair in enumerate(protein_pairs):
        protein_pair.features = calculate_features(protein_pair, feature_list)
        progress_bar(i, len(protein_pairs))

    print('Calculated features for training protein pairs.')
