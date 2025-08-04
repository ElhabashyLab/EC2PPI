class Protein:
    """Class representing a protein."""
    def __init__(self, uniprot_id: str):
        self.uid = uniprot_id
        self.sequence = None
