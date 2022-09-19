from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import hydra
import numpy as np
import spacy
import torch
from spacy.matcher import Matcher

from consec.src.consec_dataset import ConsecDefinition, ConsecSample
from consec.src.consec_tokenizer import ConsecTokenizer
from consec.src.disambiguation_corpora import DisambiguationInstance
from consec.src.pl_modules import ConsecPLModule
from consec.src.scripts.model.predict import predict
from consec.src.sense_inventories import SenseInventory, WordNetSenseInventory
from consec.src.utils.wsd import pos_map

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)


@dataclass
class Instance:
    id: str
    tokens: List[str]
    target_position: int
    lemma: str
    pos_tag: str


@dataclass
class InstancePrediction:
    id: str
    candidate_senses: List[str]
    probabilities: List[float]


def custom_predict(
    instance_iterator: Iterator[Instance],
    module: ConsecPLModule,
    sense_inventory: SenseInventory,
    tokenizer: ConsecTokenizer,
) -> Iterator[InstancePrediction]:

    consec_samples: List[ConsecSample] = []
    candidate_senses: Dict[str, List[str]] = {}

    for instance in instance_iterator:
        instance_candidate_senses = sense_inventory.get_possible_senses(
            lemma=instance.lemma, pos=instance.pos_tag
        )
        instance_candidate_definitions = [
            ConsecDefinition(sense_inventory.get_definition(s), instance.lemma)
            for s in instance_candidate_senses
        ]
        consec_sample = ConsecSample(
            sample_id=instance.id,
            position=instance.target_position,
            disambiguation_context=[
                DisambiguationInstance("d0", "s0", "i0", t, "", "", None)
                for t in instance.tokens
            ],
            candidate_definitions=instance_candidate_definitions,
            gold_definitions=None,
            context_definitions=[],
            in_context_sample_id2position={
                instance.id: instance.target_position
            },
            disambiguation_instance=None,
            kwargs={},
        )

        consec_samples.append(consec_sample)
        candidate_senses[instance.id] = instance_candidate_senses

    prediction_iterator = predict(
        module=module,
        tokenizer=tokenizer,
        samples=consec_samples,
        text_encoding_strategy="simple-with-linker",
    )

    for consec_sample, probs in prediction_iterator:
        instance_prediction = InstancePrediction(
            id=consec_sample.sample_id,
            candidate_senses=candidate_senses[consec_sample.sample_id],
            probabilities=probs,
        )

        yield instance_prediction


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


def init_module_and_tokenizer(
    model_checkpoint_path: str, device: str
) -> Tuple[ConsecPLModule, ConsecTokenizer]:
    print(model_checkpoint_path)
    module = ConsecPLModule.load_from_checkpoint(model_checkpoint_path)
    module.to(torch.device(device if device != "-1" else "cpu"))
    module.freeze()
    module.sense_extractor.evaluation_mode = True

    # load tokenizer
    tokenizer = hydra.utils.instantiate(
        module.hparams.tokenizer.consec_tokenizer
    )

    return module, tokenizer


def main() -> None:
    root = Path(__file__).absolute().parent

    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="-1")
    parser.add_argument(
        "--model_checkpoint_path",
        type=Path,
        default=root / "consec" / "checkpoints" / "consec_wngt_best.ckpt",
    )
    parser.add_argument(
        "--wn_candidates_path",
        type=Path,
        default=root / "consec" / "data" / "candidatesWN30.txt",
    )
    parser.add_argument("--data_path", type=Path)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    instance_iterator = read_se23(args.data_path)

    module, tokenizer = init_module_and_tokenizer(
        args.model_checkpoint_path, args.device
    )
    wn_sense_inventory = WordNetSenseInventory(args.wn_candidates_path)

    with open(args.output, "w") as f:
        for instance_prediction in custom_predict(
            instance_iterator, module, wn_sense_inventory, tokenizer
        ):
            idx = np.argmax(instance_prediction.probabilities)

            predicted_sense = instance_prediction.candidate_senses[idx]
            id = instance_prediction.id
            f.write(f"{id}\t{predicted_sense}\n")


if __name__ == "__main__":
    main()
