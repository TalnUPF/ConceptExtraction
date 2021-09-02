# -*- coding: utf-8 -*-
import os
import sys
import random
import shutil
import re
import string
from stop_words import get_stop_words
from datetime import datetime
from collections import defaultdict
from itertools import chain

import logging
logging.basicConfig(filename='debug_logging.log', level=logging.INFO)

sys.path.insert(0, 'OpenNMT-py')
import translate

from src.text_processor import Token
from src.concept_extractor import Concept


class ConceptExtractor:
    """
    Core class where extract_concepts is the main function which takes tokenized texts and a set of flags as input
    and returns lists of concepts grouped according to the grammatical form (by default, only noun phrases and numbers
    that are combined in the same list).
    """
    def __init__(self, lang):
        self.lang = lang
        self.sentences = []
        self.sentence_to_text_dict = {}
        self.src_sequence_list = []
        self.sentence_id_list = []
        self.src_tokens_with_positions = []
        self.parser = translate.get_parser()

        self.utf_chars = [
            u'\u2014', u'\u2013', u'\uFFFD',
            u'\u2026', u'\u2665', u'\ud83d',
            u'\u0306', u'\u201E', u'\u200B',
            u'\u201C', u'\u2116', u'\u2022',
            u'\ud83c', u'\u201D', u'\ude02',
            u'\u2019', u'\u00BB', u'\u00AB',
            u'\u2039', u'\u276E', u'\u003C',
            u'\u16B2', u'\u1438', u'\u02C2',
            u'\u00ab', u'\u203A', u'\u276F',
            u'\u02C3', u'\u1433', u'\u003E'
            ]
        self.stop_words = get_stop_words(self.lang)
        self.stop_words.extend(self.utf_chars)

    def __enter__(self, ):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.lang
        del self.stop_words
        del self.utf_chars
        del self.sentences
        del self.sentence_to_text_dict
        del self.src_sequence_list
        del self.sentence_id_list
        del self.src_tokens_with_positions
        del self.parser

    """
    Format of the input:
        A dictionary with a field "sentence_list"
        where "sentence_list" is a list of dictionaries with fields "token_list" and "text_id" (optional)
        and "token_list" is a list of dictionaries with fields "token", "tag", "beg_offset", "end_offset"
    """
    def read_input(self, tokenized_texts):
        self.sentences = []
        self.sentence_to_text_dict = {}
        for sent_num, sentence_meta in enumerate(tokenized_texts["sentence_list"]):
            token_list = [
                Token(token=tok["token"], postag=tok["tag"] if "tag" in tok else "", beg_offset=tok["beg_offset"],
                      end_offset=tok["end_offset"]) for tok in sentence_meta["token_list"]]
            self.sentences.append(token_list)
            if "text_id" in sentence_meta:
                self.sentence_to_text_dict[sent_num] = sentence_meta["text_id"]

    def prepare_input_sequences(self,):
        pointer_generator_folder = "tmp_seq2seq_%s" % (
                    datetime.now().strftime("%m%d%Y%H%M%S") + str(random.randint(1000, 9999)))
        if not os.path.exists(pointer_generator_folder):
            os.makedirs(pointer_generator_folder)

        window_size = 20
        shift_size = 5
        self.src_sequence_list = []
        self.sentence_id_list = []
        self.src_tokens_with_positions = []
        for sent_id, sentence in enumerate(self.sentences):
            if len(sentence) < 1:
                continue
            last_token_index = min(len(sentence), window_size)
            window_tokens = sentence[:last_token_index]
            src_sequence = " ".join(map(lambda x: re.sub("\s", "-", x.token)+" "+x.postag, window_tokens))
            self.src_sequence_list.append(src_sequence)
            self.sentence_id_list.append(sent_id)
            self.src_tokens_with_positions.append(
                [[wtok.token, dtoc] for dtoc, wtok in enumerate(window_tokens)])
            while last_token_index < len(sentence):
                first_token_index = last_token_index - shift_size
                last_token_index = min(len(sentence), last_token_index + window_size)
                window_tokens = sentence[first_token_index:last_token_index]
                src_sequence = " ".join(map(lambda x: re.sub("\s","-",x.token)+" "+x.postag, window_tokens))
                self.src_sequence_list.append(src_sequence)
                self.sentence_id_list.append(sent_id)
                self.src_tokens_with_positions.append(
                    [[wtok.token, first_token_index + dtoc] for dtoc, wtok in enumerate(window_tokens)])
        with open(os.path.join(pointer_generator_folder, "src_sequences.txt"), "wb") as fout:
            fout.write(("\n".join(self.src_sequence_list) + "\n").encode("utf8"))
        with open(os.path.join(pointer_generator_folder, "sentence_id_list.txt"), "wb") as fout:
            fout.write(("\n".join(list(map(lambda x: str(x), self.sentence_id_list)))+"\n").encode("utf8"))
        return pointer_generator_folder

    def run_opennmt_translate(self, pointer_generator_folder):
        translate_args = list()
        translate_args.append(
            {"model": os.path.join("models", self.lang, "model_dsa_60_0_20K_2l_step_18000.pt"),
             "src_file": os.path.join(pointer_generator_folder, "src_sequences.txt"),
             "output_file": os.path.join(pointer_generator_folder, "predicted_by_model_1.txt"),
             "all_concepts_lowercase": False,
             "do_fix_sequences": True
             })
        translate_args.append(
            {"model": os.path.join("models", self.lang, "model_dsa_dbpedia_spotlight10_100K_3l_step_80000.pt"),
             "src_file": os.path.join(pointer_generator_folder, "src_sequences.txt"),
             "output_file": os.path.join(pointer_generator_folder, "predicted_by_model_2.txt"),
             "all_concepts_lowercase": False,
             "do_fix_sequences": True
             })
        for params in translate_args:
            opt = self.parser.parse_args(["-model", params["model"],
                                          "-src", params["src_file"],
                                          "-output", params["output_file"],
                                          "-replace_unk"])  # "-gpu", 0
            translate.main(opt)
        return translate_args

    @staticmethod
    def align_generated_sequences(src_tokens, generated_concept_candidates, all_concepts_lowercase=False,
                                  do_fix_sequences=True):
        concept_candidates_aligned = []
        gen_conc_i = 0
        while gen_conc_i < len(generated_concept_candidates):
            try:
                conc_text = generated_concept_candidates[gen_conc_i][0]
                conc_type = generated_concept_candidates[gen_conc_i][1]
                if do_fix_sequences:
                    conc_text = re.sub(b" of ((the)|(a)|(an))$", b"",
                                       re.sub(b" of$", b"", re.sub(b" 's$", b"", conc_text))).strip()
                    skip_symbols = [u"‹".encode("utf8"), u"«".encode("utf8"), u"»".encode("utf8"), u"›".encode("utf8"),
                                    u"·".encode("utf8"), u"’".encode("utf8"), u"”".encode("utf8"), u"“".encode("utf8"),
                                    u"…".encode("utf8"), u"—".encode("utf8"),
                                    u"‘".encode("utf8"), u"●".encode("utf8"), u"❍".encode("utf8")]
                    for symbol in skip_symbols:
                        conc_text = re.sub(symbol, b"", conc_text).strip()
                    conc_text = re.sub(b"\|", b"", conc_text).strip()
                tokens_str_list = conc_text.split(b" ")
                tokens_str_list = [b"\"" if tok == b"''" or tok == b"``" else tok for tok in tokens_str_list]
                src_tok_i = 0
                while src_tok_i < len(src_tokens):
                    if all_concepts_lowercase:
                        if tokens_str_list == list(map(lambda x: x[0].encode("utf8").lower(),
                                                       src_tokens[src_tok_i:src_tok_i + len(tokens_str_list)])):
                            concept_candidates_aligned.append(tuple([src_tokens[src_tok_i][1],
                                                                     src_tokens[src_tok_i + len(tokens_str_list) - 1][
                                                                         1], conc_type]))
                    else:
                        if tokens_str_list == list(map(lambda x: x[0].encode("utf8"),
                                                       src_tokens[src_tok_i:src_tok_i + len(tokens_str_list)])):
                            concept_candidates_aligned.append(tuple([src_tokens[src_tok_i][1],
                                                                     src_tokens[src_tok_i + len(tokens_str_list) - 1][
                                                                         1], conc_type]))
                    src_tok_i += 1
                gen_conc_i += 1
            except:
                gen_conc_i += 1
        return concept_candidates_aligned

    def prepare_concept_candidates(self, translate_args, include_numbers, include_adj_and_verbs):
        sentence_to_concepts_dict = {}
        for params in translate_args:
            generated_seqs = open(params["output_file"], "rb").read().splitlines()
            for seq_num, sent_id in enumerate(self.sentence_id_list):
                if not sent_id in sentence_to_concepts_dict:
                    sentence_to_concepts_dict[sent_id] = {}
                gen_candidates = re.sub(b"\*\*", b"*", generated_seqs[seq_num].strip()).split(b" * ")
                src_tokens_with_pos_list = self.src_sequence_list[seq_num].strip().split(" ")
                if len(src_tokens_with_pos_list) % 2 == 1:
                    src_tokens_with_pos_list = []
                    logging.warning("Something wrong with src_tokens_with_pos_list in line %d" % (seq_num), sent_id,
                                    params)
                gen_candidates.extend([src_tokens_with_pos_list[dtok].encode("utf8") for dtok in
                                       range(0, len(src_tokens_with_pos_list), 2) if
                                       (
                                               src_tokens_with_pos_list[dtok + 1][0] == "N" and
                                               not src_tokens_with_pos_list[dtok + 1] in ["NUM"]
                                       )
                                       or (include_numbers and src_tokens_with_pos_list[dtok + 1] in ["CD", "NUM"])])
                gen_candidates = list(map(lambda x: [x, "NOUN"], gen_candidates))
                if include_adj_and_verbs:
                    gen_candidates = list(chain.from_iterable([gen_candidates,
                                                               [[src_tokens_with_pos_list[dtok].encode("utf8"),
                                                                 "ADJ"] for dtok in
                                                                range(0, len(src_tokens_with_pos_list), 2) if
                                                                (src_tokens_with_pos_list[dtok + 1][0] == "J" or
                                                                 src_tokens_with_pos_list[dtok + 1] in ["ADJ"])],
                                                               [[src_tokens_with_pos_list[dtok].encode("utf8"),
                                                                 "VERB"] for dtok in
                                                                range(0, len(src_tokens_with_pos_list), 2) if
                                                                (src_tokens_with_pos_list[dtok + 1][0] == "V" or
                                                                 src_tokens_with_pos_list[dtok + 1] in ["VERB"])]]))
                if not "concept_candidates" in sentence_to_concepts_dict[sent_id]:
                    sentence_to_concepts_dict[sent_id]["concept_candidates"] = self.align_generated_sequences(
                        self.src_tokens_with_positions[seq_num], gen_candidates, params["all_concepts_lowercase"],
                        params["do_fix_sequences"])
                else:
                    sentence_to_concepts_dict[sent_id]["concept_candidates"].extend(
                        self.align_generated_sequences(self.src_tokens_with_positions[seq_num], gen_candidates,
                                                       params["all_concepts_lowercase"], params["do_fix_sequences"]))
        return sentence_to_concepts_dict

    def find_proper_end_token(self, concept_candidate_postag_sequence, concept_candidate_type):
        """
        Only n-grams ending with nouns or numbers are treated as noun phrase concepts.
        """
        end_token_position = len(concept_candidate_postag_sequence) - 1
        if self.lang == "en" and concept_candidate_type == "NOUN":
            end_token_position = -1
            for token_num, postag in enumerate(reversed(concept_candidate_postag_sequence)):
                if postag[0] == "N" or postag == "CD":
                    end_token_position = len(concept_candidate_postag_sequence) - 1 - token_num
                    break
        return end_token_position

    def is_concept_text_alphanumeric(self, concept_text):
        concept_text = concept_text.lower()
        concept_text_clear = concept_text
        text_not_empty = False
        for wsym in self.utf_chars:
            concept_text_clear = re.sub(wsym, u"", concept_text_clear).strip()
        if sys.version_info[0] == 2:
            text_not_empty = (not concept_text in self.stop_words) and (not concept_text_clear in self.stop_words) and (
                        concept_text_clear.encode("utf8").translate(
                            string.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(" ", "") != "")
        else:
            text_not_empty = (not concept_text in self.stop_words) and (not concept_text_clear in self.stop_words) and (
                        concept_text_clear.translate(
                            str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(" ", "") != "")
        return text_not_empty and not re.search(u"@|#|(http)", concept_text)

    def get_non_overlapping_concepts(self, sentence_to_concepts_dict):
        for sent_id in sentence_to_concepts_dict:
            concept_candidates_list = sorted(list(set(sentence_to_concepts_dict[sent_id]["concept_candidates"])),
                                             key=lambda x: x[0] * 1000000 - x[1] - (0.1 if x[2] == 'NOUN' else 0))
            already_used_indices = set()
            concepts_list = []
            for concept_candidate in concept_candidates_list:
                begin_index = concept_candidate[0]
                end_index = concept_candidate[1]
                concept_candidate_postag_sequence = [self.sentences[sent_id][tok_i].postag for tok_i in
                                                     range(begin_index, end_index + 1)]
                concept_candidate_type = concept_candidate[2]
                end_token_position = self.find_proper_end_token(concept_candidate_postag_sequence,
                                                                concept_candidate_type)
                end_index = begin_index + end_token_position
                if end_token_position >= 0 and self.is_concept_text_alphanumeric(" ".join(
                        [self.sentences[sent_id][tok_i].token for tok_i in range(begin_index, end_index + 1)])):
                    if all((not tok_i in already_used_indices for tok_i in range(begin_index, end_index + 1))):
                        already_used_indices.update(list(range(begin_index, end_index + 1)))
                        concepts_list.append(Concept(tokens_list=[self.sentences[sent_id][tok_i] for tok_i in
                                                                  range(begin_index, end_index + 1)],
                                                     type_of_concept=concept_candidate_type,
                                                     end_index=end_index))
            sentence_to_concepts_dict[sent_id]["concepts"] = concepts_list

        concepts_by_type_output = defaultdict(list)
        for sent_id in sorted(sentence_to_concepts_dict.keys()):
            for concept in sentence_to_concepts_dict[sent_id]["concepts"]:
                concept_details = {"concept": " ".join([tok.token for tok in concept.tokens]),
                                   "postags": " ".join([tok.postag for tok in concept.tokens]),
                                   "begin": concept.tokens[0].beg_offset,
                                   "end": concept.tokens[-1].end_offset,
                                   "type": concept.type,
                                   "next_tag": self.sentences[int(sent_id)][concept.end_index + 1].postag if len(self.sentences[int(sent_id)])>concept.end_index + 1 else "EOS_tag",
                                   "next_word": self.sentences[int(sent_id)][concept.end_index + 1].token if len(self.sentences[int(sent_id)])>concept.end_index + 1 else "EOS_token",
                                   "sent_id": sent_id}
                if len(self.sentence_to_text_dict) > 0:
                    concept_details.update({"text_id": self.sentence_to_text_dict[sent_id]})
                concepts_by_type_output[concept.type].append(concept_details)
        concepts_by_type_output["concepts"] = concepts_by_type_output["NOUN"][:]
        del concepts_by_type_output["NOUN"]
        return dict(concepts_by_type_output)

    def extract_concepts(self, tokenized_texts, keep_input_output_sequences,
                         include_numbers, include_adj_and_verbs):
        self.read_input(tokenized_texts)
        pointer_generator_folder = self.prepare_input_sequences()
        translate_args = self.run_opennmt_translate(pointer_generator_folder)
        sentence_to_concepts_dict = self.prepare_concept_candidates(translate_args,
                                                                    include_numbers,
                                                                    include_adj_and_verbs)
        concepts_by_type_dict = self.get_non_overlapping_concepts(sentence_to_concepts_dict)

        if not keep_input_output_sequences:
            if os.path.exists(pointer_generator_folder):
                try:
                    shutil.rmtree(pointer_generator_folder)
                except OSError:
                    logging.error("The temporary folder has not been removed")

        return concepts_by_type_dict
