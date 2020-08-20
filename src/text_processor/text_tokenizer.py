import re

from src.text_processor import TokenizedText


class TextTokenizer:
    """
    Creates an input in an appropriate form for ConceptExtractor.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_tagged_sentences_from_file(path_to_text_per_line_file):
        """Splits each text in a file into sentences and extracts tokens from them."""
        output_dict = dict()
        output_dict["sentence_list"] = []
        for text_id, text_str in enumerate(open(path_to_text_per_line_file, "rb").read().splitlines()):
            text_str = text_str.decode("utf8").strip()
            text_str = re.sub("``", "\"", text_str)
            text_str = re.sub("''", "\"", text_str)
            tokenized_text = TokenizedText()
            tokenized_text.create_from_text(text_str)
            output_dict["sentence_list"].extend([{"token_list": [{"token": token.token, "tag": token.postag,
                                                                  "beg_offset": token.beg_offset,
                                                                  "end_offset": token.end_offset
                                                                  }
                                                                 for token in sentence], "text_id": text_id} for
                                                 sentence in tokenized_text.sentences])
        return output_dict
