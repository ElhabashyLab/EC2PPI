from src.utils.protein import Protein
class ProteinPair:
    """Class representing a pair of proteins."""
    def __init__(self, prefix: str, protein1: Protein, protein2: Protein, ec_filepath: str, af3_filepath: str = None,label: int = None, pairwise_identity: float = None):
        self.prefix = prefix
        self.protein1 = protein1
        self.protein2 = protein2
        self.ec_filepath = ec_filepath
        self.af3_filepath = af3_filepath
        self.label = label
        self.pairwise_identity = pairwise_identity

        # Initialize additional attributes
        self.features = None