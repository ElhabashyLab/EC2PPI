from src.utils.protein import Protein
class ProteinPair:
    """Class representing a pair of proteins."""
    def __init__(self, protein1: Protein, protein2: Protein):
        self.protein1 = protein1
        self.protein2 = protein2

        # future attributes
        self.af3_directory = None