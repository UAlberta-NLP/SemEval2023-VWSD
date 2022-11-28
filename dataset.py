from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List
import fire
from bs4 import BeautifulSoup
import os

import spacy
from spacy.matcher import Matcher
from consec.src.utils.wsd import pos_map

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
wsd_pos_map = {"v": "VERB", "n": "NOUN", "a": "ADJ", "r": "ADV"}


@dataclass
class Instance:
    id: str
    tokens: List[str]
    target_position: int
    lemma: str
    pos_tag: str


def prepare_text(text: str, lemma: str, id: str) -> Instance:
    doc = nlp(text)

    patterns = [[{"LOWER": lemma.lower()}], [{"LEMMA": lemma.lower()}]]
    matcher.add("lemma_match", patterns)
    matches = matcher(doc)

    assert len(matches) >= 1
    match_id, start, end = matches[0]
    assert end == (start + 1)
    tokens = [t.text for t in doc]
    pos = doc[start].pos_
    target_position = start

    matcher.remove("lemma_match")

    instance = Instance(
        id=id,
        tokens=tokens,
        target_position=target_position,
        lemma=lemma,
        pos_tag=pos_map[pos],
    )

    return instance


def read_se23(path: Path) -> Iterator[Instance]:
    with open(path, "r") as f:
        for line_number, line in enumerate(f):
            data = line.strip().split("\t")
            lemma, text = data[0], data[1]

            instance = prepare_text(text, lemma, str(line_number))

            yield instance


def build_raganato_dataset(dataset_path: str, output_folder: str):
    soup = BeautifulSoup("", "xml", preserve_whitespace_tags=["wf", "instance"])
    corpus_tag = soup.new_tag("corpus", lang="en", source="source")
    text_tag = soup.new_tag("text", id="XXX")
    corpus_tag.append(text_tag)

    for instance in read_se23(dataset_path):
        sentence_tag = soup.new_tag("sentence", id=instance.id)
        text_tag.append(sentence_tag)

        for idx, token in enumerate(instance.tokens):
            if token is None or token == "":
                continue
            if idx == instance.target_position:
                token_tag = soup.new_tag("instance")
                token_tag["lemma"] = instance.lemma
                token_tag["pos"] = wsd_pos_map[pos_map[instance.pos_tag]]
                token_tag["id"] = instance.id
            else:
                token_tag = soup.new_tag("wf")
            token_tag.string = token
            sentence_tag.append(token_tag)

    soup.append(corpus_tag)
    dataset_name = f"{Path(output_folder).name}.data.xml"
    labels_name = f"{Path(output_folder).name}.gold.key.txt"

    os.makedirs(output_folder, exist_ok=True)
    with open(Path(output_folder) / dataset_name, "w") as f:
        f.write(soup.prettify())

    with open(Path(output_folder) / labels_name, "w") as f:
        ...


if __name__ == "__main__":
    fire.Fire()
