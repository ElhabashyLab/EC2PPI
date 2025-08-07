class Protein:
    """Class representing a protein."""
    def __init__(self, uniprot_id: str, n_eff: float = None, n_eff_l: float = None, sequence_length: int = None, bit_score: float = None):
        self.uid = uniprot_id
        self.n_eff = n_eff
        self.n_eff_l = n_eff_l
        self.sequence_length = sequence_length
        self.bit_score = bit_score


