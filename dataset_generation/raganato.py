from typing import Iterator, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass
from bs4 import BeautifulSoup, Tag


@dataclass
class WSDToken:
    text: str
    lemma: str
    pos: str
    id: Optional[str]

    @staticmethod
    def from_xml_tag(tag: Tag) -> "WSDToken":
        token = WSDToken(
            text=tag.text, lemma=tag["lemma"], pos=tag["pos"], id=tag.get("id", None)
        )

        return token


@dataclass
class WSDSentence:
    tokens: List[WSDToken]
    target_ids: List[int]

    @staticmethod
    def from_xml_tag(tag: Tag) -> "WSDSentence":
        tokens = [
            WSDToken.from_xml_tag(child) for child in tag if child.name is not None
        ]
        target_ids = [idx for idx, t in enumerate(tokens) if t.id is not None]

        sentence = WSDSentence(tokens=tokens, target_ids=target_ids)

        return sentence


class WSDDataset:
    def __init__(self, instances: Iterator[WSDSentence]) -> None:
        self.instances = instances

    @staticmethod
    def from_xml(path: Union[str, Path]) -> "WSDDataset":
        def _generate_instances() -> Iterator[WSDSentence]:
            with open(path, "r") as f:
                soup = BeautifulSoup(f, "xml")

            for sentence_tag in soup.find_all("sentence"):
                yield WSDSentence.from_xml_tag(sentence_tag)

        return WSDDataset(instances=_generate_instances())

    def __iter__(self) -> Iterator[WSDSentence]:
        yield from self.instances
