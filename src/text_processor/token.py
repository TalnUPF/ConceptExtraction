class Token:
    """
    A token includes its surface form, PoS-tag, and offsets in the text.
    """
    def __init__(self, token: str, postag: str, beg_offset: int, end_offset: int):
        self.token = token
        self.postag = postag if postag else "UNK"
        self.beg_offset = beg_offset
        self.end_offset = end_offset
