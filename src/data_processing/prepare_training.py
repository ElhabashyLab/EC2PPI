def prepare_test_and_training_data(data, portion=0.1):
    """
    Prepares the training data for a machine learning model.

    :param data: List of protein_pairs
    :param portion: Portion of data to be used for testing (default is 0.1, i.e., 10% for testing and 90% for training)
    :return: List of proteins for training and list of proteins for testing
    """

