from argparse import ArgumentParser
from pathlib import Path
from raganato import WSDDataset, read_raganato_labels
import spacy
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.lang.en.stop_words import STOP_WORDS
import re

nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS | {"``"} - {"call", "former", "serious", "side"}


def remove_stop_words(span):
    return [
        t.lower_
        for t in span
        if t.lower_ not in stop_words and t.pos_ not in {"PUNCT", "NUM", "SYM"}
    ]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--xml_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--sense_keys_path", type=Path, required=True)

    args = parser.parse_args()

    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r"\S+").match)

    dataset = WSDDataset.from_xml(args.xml_path)
    sense_keys = read_raganato_labels(args.sense_keys_path)

    with open(args.output_path, "w") as f:
        for sentence in dataset:
            doc = nlp(Doc(nlp.vocab, [t.text for t in sentence.tokens]))
            noun_chunks = doc.noun_chunks

            for noun_chunk in noun_chunks:
                for target_idx in sentence.target_ids:
                    if (
                        (sentence.tokens[target_idx].pos != "NOUN")
                        or not (noun_chunk.start <= target_idx < noun_chunk.end)
                        or len(noun_chunk) >= 4
                        or sentence.tokens[target_idx].text.isupper()
                    ):
                        continue
                    candidate = remove_stop_words(noun_chunk)
                    if len(candidate) <= 1:
                        continue

                    context = " ".join(candidate)
                    target_token = sentence.tokens[target_idx]
                    target_word = target_token.text.lower()
                    sense_key = sense_keys[target_token.id]

                    if len(target_word.split(" ")) > 1:
                        data = [
                            target_word,
                            context,
                            str(len(target_word.split(" ")) - 1),
                            sense_key,
                        ]
                    else:
                        try:
                            idx = candidate.index(target_word.lower())
                        except ValueError:
                            print(f"Could not find target index for {target_token.id} {target_word}")
                            continue
                        data = [
                            target_word,
                            context,
                            str(idx),
                            sense_key,
                        ]
                    f.write("\t".join(data) + "\n")


if __name__ == "__main__":
    main()
