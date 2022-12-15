from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)

    args = parser.parse_args()

    with open(args.output_path, "w+") as f:
        sense_keys = set(tuple(line.strip().split("\t")) for line in f.readlines())

    with open(args.input_path, "r") as f:
        for line in f:
            *other, lemma, sense_key = line.strip().split("\t")
            sense_keys.add((lemma, sense_key))

    with open(args.output_path, "w") as f:
        for lemma, sense_key in sense_keys:
            f.write(f"{lemma}\t{sense_key}\n")


if __name__ == "__main__":
    main()
