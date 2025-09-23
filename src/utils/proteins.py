class Protein:
    """Class representing a protein."""
    def __init__(self, uniprot_id: str, n_eff: float = None, n_eff_l: float = None, sequence_length: int = None, bit_score: float = None):
        self.uid = uniprot_id
class ProteinPair:
    """Class representing a pair of proteins."""
    def __init__(self, prefix: str, protein1: Protein, protein2: Protein,label: int = None, ec_filepath: str = None, af3_directory: str = None, custom_features: dict = None):
        self.prefix = prefix
        self.protein1 = protein1
        self.protein2 = protein2

        self.label = label
        self.ec_filepath = ec_filepath
        self.af3_directory = af3_directory
        self.custom_features = custom_features
        self.features = None