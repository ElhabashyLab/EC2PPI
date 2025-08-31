class Protein:
    """Class representing a protein."""
    def __init__(self, uniprot_id: str, n_eff: float = None, n_eff_l: float = None, sequence_length: int = None, bit_score: float = None):
        self.uid = uniprot_id
        self.n_eff = n_eff
        self.n_eff_l = n_eff_l
        self.sequence_length = sequence_length
        self.bit_score = bit_score
class ProteinPair:
    """Class representing a pair of proteins."""
    def __init__(self, prefix: str, protein1: Protein, protein2: Protein, ec_filepath: str = None, af3_directory: str = None,label: int = None, pairwise_identity: float = None):
        self.prefix = prefix
        self.protein1 = protein1
        self.protein2 = protein2

        self.label = label
        self.pairwise_identity = pairwise_identity

        # Initialize additional attributes
        self.ec_filepath = ec_filepath
        self.af3_directory = af3_directory
        self.features = None