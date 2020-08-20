import re
import nltk
import nltk.data

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize

from src.text_processor import Token


class TokenizedText:
    """
    Contains a list of sentences where a sentence is a list of tokens (instances of a class Token)
    """
    
    def __init__(self, ):
        self.sentences = []
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    @staticmethod
    def get_token_offsets(tags, sentence_str, text_str, sent_offset):
        offset = 0
        token_offsets = []
        if text_str != "":
            sent_offset = text_str.find(tags[0][0], sent_offset) - sentence_str.find(tags[0][0], offset)
        for tag in tags:
            offset = sentence_str.find(tag[0], offset)
            token_offsets.append([sent_offset + offset, sent_offset + offset + len(tag[0])])
            offset += len(tag[0])
        return token_offsets

    def create_from_text(self, text_str):
        detected_sentences = self.sent_detector.tokenize(text_str)
        sent_offset = 0
        for sentnum, sentence_str in enumerate(detected_sentences):
            sentence_str = re.sub("``", "\"", sentence_str)
            sentence_str = re.sub("''", "\"", sentence_str)
            tokens = word_tokenize(sentence_str)
            tags = nltk.pos_tag(tokens)
            tags = [("\"", "\"") if tg[0] == "''" or tg[0] == "``" else tg for tg in tags]
            token_offsets = self.get_token_offsets(tags, sentence_str, text_str, sent_offset)
            token_list = [Token(token=tags[i][0], postag=tags[i][1], beg_offset=token_offsets[i][0],
                                end_offset=token_offsets[i][1]) for i in range(len(tags))]
            self.sentences.append(token_list)
            sent_offset = text_str.find(tags[0][0], sent_offset) - sentence_str.find(tags[0][0], 0) + len(sentence_str)
