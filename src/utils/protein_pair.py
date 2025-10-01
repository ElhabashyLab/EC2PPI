class ProteinPair:
    """Class representing a pair of proteins."""
    def __init__(self, prefix: str, uid1: str, uid2: str, label: int = None, ec_filepath: str = None, af3_directory: str = None, custom_features: dict = None):
        self.prefix = prefix
        self.uid1 = uid1
        self.uid2 = uid2

        self.label = label
        self.ec_filepath = ec_filepath
        self.af3_directory = af3_directory
        self.custom_features = custom_features
        self.features = None