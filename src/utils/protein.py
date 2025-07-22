class Protein:
    def __init__(self, uniprot_id: str ,sequence: str= None):
        self._uid = uniprot_id
        self._sequence = sequence
