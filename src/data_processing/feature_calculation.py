import pandas as pd
import json
import os
import numpy as np
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

def features_from_af3_directory(af3_directory: str, feature_list: list):
    """Extract features from the AF3 directory.

    :param af3_directory: Path to the AF3 directory.
    :param feature_list: List of features to extract.
    :return: List of features extracted from the AF3 directory.
    """

    all_files = [entry.path for entry in os.scandir(af3_directory) if (entry.is_file()) or (entry.is_dir())]

    summary_conf = None
    confidences = None

    for file in all_files:
        if file.endswith('_summary_confidences.json'):
            with open(file, 'r') as f:
                summary_conf = json.load(f)
        elif file.endswith('_confidences.json'):
            with open(file, 'r') as f:
                confidences = json.load(f)

    if summary_conf is None or confidences is None:
        raise ValueError(f'Could not find summary_confidences.json or confidences.json in {af3_directory}')


    features = []

    if 'iptm' in feature_list:
        features.append(summary_conf['iptm'])

    if 'fraction_disordered' in feature_list:
        features.append(summary_conf['fraction_disordered'])

    if 'plddt_mean' in feature_list:
        plddts = pd.Series(confidences['atom_plddts'])
        features.append(plddts.mean())

    # only if usage of pae between the two proteins
    if any(feature.startswith('pae_') for feature in feature_list):
        pae = np.array(confidences['pae'])
        token_chain_ids = confidences['token_chain_ids']
        ids , counts = np.unique(token_chain_ids, return_counts=True)
        n = counts[0] if token_chain_ids[0] == ids[0] else counts[1]

        sub_matrix_pae_21 = pae[:n,n:]
        sub_matrix_pae_12 = pae[n:,:n].transpose()

        sub_matrix_pae = np.concatenate((sub_matrix_pae_21, sub_matrix_pae_12), axis=0)
        sub_series_pae = pd.Series(sub_matrix_pae.flatten())

        if 'pae_mean' in feature_list:
            features.append(sub_series_pae.mean())

        if 'pae_std' in feature_list:
            features.append(sub_series_pae.std())

        if 'pae_median' in feature_list:
            features.append(np.median(sub_series_pae))

        if 'pae_iqr' in feature_list:
            q75, q25 = np.percentile(sub_series_pae, [75, 25])
            features.append(q75 - q25)

        if 'pae_max' in feature_list:
            features.append(sub_series_pae.max())

        if 'pae_skewness' in feature_list:
            features.append(sub_series_pae.skew())

        if 'pae_kurtosis' in feature_list:
            features.append(sub_series_pae.kurtosis())

    # only if usage of contact probs between the two proteins
    if any(feature.startswith('contact_probs_') for feature in feature_list):
        token_chain_ids = confidences['token_chain_ids']
        ids , counts = np.unique(token_chain_ids, return_counts=True)
        n = counts[0] if token_chain_ids[0] == ids[0] else counts[1]

        contact_probs = np.array(confidences['contact_probs'])
        sub_matrix_cb = contact_probs[n:,:n]
        sub_series_cb = pd.Series(sub_matrix_cb.flatten())

        if 'contact_probs_mean' in feature_list:
            features.append(sub_series_cb.mean())

        if 'contact_probs_std' in feature_list:
            features.append(sub_series_cb.std())

        if 'contact_probs_median' in feature_list:
            features.append(np.median(sub_series_cb))

        if 'contact_probs_iqr' in feature_list:
            q75, q25 = np.percentile(sub_series_cb, [75, 25])
            features.append(q75 - q25)

        if 'contact_probs_max' in feature_list:
            features.append(sub_series_cb.max())

        if 'contact_probs_skewness' in feature_list:
            features.append(sub_series_cb.skew())

        if 'contact_probs_kurtosis' in feature_list:
            features.append(sub_series_cb.kurtosis())




    return features

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

    # calculate features from AF3 file
    features.extend(features_from_af3_directory(protein_pair.af3_directory, feature_list))

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
