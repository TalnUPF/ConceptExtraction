class Concept:
    """
    Concept is a sequence of tokens (instances of a class Token).
    The type corresponds to a grammatical form of a concept:
        NOUN: noun phrases (including individual nouns) and numbers,
        ADJ: adjectives,
        ADV: adverbs,
        VERB: verbs.
    """
    def __init__(self, tokens_list, type_of_concept, end_index):
        self.tokens = tokens_list
        self.type = type_of_concept
        self.end_index = end_index
