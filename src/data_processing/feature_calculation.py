import pandas as pd
import json
import os
import numpy as np
from src.utils.protein_pair import ProteinPair
from src.utils.print import progress_bar
from joblib import Parallel, delayed


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

        # extract sub-matrix of PAE between the two proteins
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

    # calculate features from EC file
    if protein_pair.ec_filepath:
        features.extend(features_from_ec_file(protein_pair.ec_filepath, feature_list))

    # calculate features from AF3 file
    if protein_pair.af3_directory:
        features.extend(features_from_af3_directory(protein_pair.af3_directory, feature_list))

    # add custom features if available
    if protein_pair.custom_features:
        for feature in feature_list:
            if feature in protein_pair.custom_features:
                features.append(protein_pair.custom_features[feature])

    return features

def calculate_all_features(protein_pairs, params, save_features: bool = True):
    """Calculate features for all protein pairs in the list.

    :param protein_pairs: List of ProteinPair objects.
    :param params: Dictionary containing parameters, including the feature list.
    :return: None, modifies the protein_pairs in place.
    """
    print('Calculating all features...')
    feature_list = params['feature_list']
    #for i, protein_pair in enumerate(protein_pairs):
    #    protein_pair.features = calculate_features(protein_pair, feature_list)
    #    progress_bar(i, len(protein_pairs))

    # Parallel processing
    def _compute(pair, fl):
        return calculate_features(pair, fl)

    all_features_in_order = Parallel(n_jobs=-1)(
        delayed(_compute)(pair, feature_list) for pair in protein_pairs
    )

    for pair, features in zip(protein_pairs, all_features_in_order):
        pair.features = features

    if save_features:
        header = ['prefix'] + feature_list + ['label']
        feature_rows = [[pair.prefix] + pair.features + [pair.label] for pair in protein_pairs]
        feature_matrix = pd.DataFrame(feature_rows, columns=header)
        feature_matrix.to_csv(os.path.join(params['export_directory'], 'calculated_features.csv'), index=False)
        print('Saved calculated features to calculated_features.csv')

    print('Calculated features for training protein pairs.')
