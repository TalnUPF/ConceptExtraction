import os
import json
import click

from src.text_processor import TextTokenizer
from src.concept_extractor import ConceptExtractor

@click.command()
@click.option('--input-file-path', '-i', required=True,
                  type=click.Path(exists=True, file_okay=True, dir_okay=False),
                  help='Path to a textual file to extract concepts from. \
                  Each line in a file is a paragraph or a text without line breaks.')
@click.option('--output-dir-path', '-odir', required=False,
                  type=click.Path(file_okay=False, dir_okay=True),
                  default='output',
                  help='Directory to store results of concept extraction.')
@click.option('--save-tokenized-texts/--omit-tokenized-texts', required=False, default=True,
                  help='Save tokenized texts with offsets in a JSON file or omit them (default=True).')
@click.option('--keep-input-output-sequences/--remove-input-output-sequences', required=False, default=False,
                  help='Keep or remove input and output sequences of pointer-generator networks (default=False)')
@click.option('--include-numbers/--ignore-numbers', required=False, default=True,
              help='Include numbers to a list of concepts of type "noun phrase" (default=True)')
@click.option('--include-adj-and-verbs/--ignore-adj-and-verbs', required=False, default=False,
              help='Return two additional lists of concepts of types "adjectives" and "verbs" (default=False)')
def main(input_file_path, output_dir_path,
         save_tokenized_texts,
         keep_input_output_sequences,
         include_numbers, include_adj_and_verbs):

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    texts_tokenized = TextTokenizer.get_tagged_sentences_from_file(input_file_path)
    if save_tokenized_texts:
        json.dump(texts_tokenized, open(f"{output_dir_path}/texts_processed.json", "w"))
    
    with ConceptExtractor(lang="en") as concept_extractor:
        concepts_extracted = concept_extractor.extract_concepts(texts_tokenized, keep_input_output_sequences,
                                                                include_numbers, include_adj_and_verbs)
        json.dump(concepts_extracted, open(f"{output_dir_path}/out.json", "w"))


if __name__ == '__main__':
    main()
