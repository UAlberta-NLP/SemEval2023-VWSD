from argparse import ArgumentParser
from pathlib import Path
import glob


def get_synsets_with_images(folder_path: Path):
    synsets_with_images = {}
    for file in folder_path.glob("*"):
        synsets_with_images[file.stem] = file.name

    return synsets_with_images


def get_candidates(path: Path):
    candidates = {}
    with path.open("r") as f:
        for line in f:
            lemma, *synset_ids_raw = line.strip().split("\t")
            lemma = lemma.lower()
            if not synset_ids_raw:
                continue
            synset_ids = set(synset_ids_raw[0].split(" "))

            candidates[lemma] = synset_ids

    return candidates


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder", type=Path, required=True)
    parser.add_argument("--image_folder", type=Path, required=True)
    parser.add_argument("--candidates_path", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)

    args = parser.parse_args()

    synsets_with_images = get_synsets_with_images(args.image_folder)
    candidate_synset_ids = get_candidates(args.candidates_path)

    output_data_path = args.output_folder / "data.txt"
    output_gold_path = args.output_folder / "gold.txt"

    unique_output = set()

    with open(output_data_path, "w") as data, open(output_gold_path, "w") as gold:
        for dataset_path in args.dataset_folder.glob("**/outputs.tsv"):
            with open(dataset_path, "r") as f:
                for line in f:
                    (
                        target_word,
                        context,
                        target_idx,
                        lemma,
                        gold_synset_id,
                    ) = line.strip().split("\t")
                    lemma = lemma.lower()
                    target_word = target_word.lower()
                    context = context.lower()


                    if (
                        gold_synset_id not in synsets_with_images
                        or lemma not in candidate_synset_ids
                        or gold_synset_id not in candidate_synset_ids[lemma]
                    ):
                        continue

                    candidate_ids = candidate_synset_ids[lemma]
                    candidate_images = [
                        synsets_with_images[id]
                        for id in candidate_ids
                        if id in synsets_with_images
                    ]

                    if len(candidate_ids) < 1:
                        continue

                    if (target_word, context) in unique_output:
                        continue

                    unique_output.add((target_word, context))

                    output_data_list = [
                        target_word,
                        context,
                        "\t".join(candidate_images),
                    ]
                    output_data = "\t".join(output_data_list) + "\n"

                    data.write(output_data)
                    gold.write(f"{synsets_with_images[gold_synset_id]}\n")


if __name__ == "__main__":
    main()
