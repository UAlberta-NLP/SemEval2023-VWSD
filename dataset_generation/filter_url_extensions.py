from argparse import ArgumentParser
from pathlib import Path
from urllib.parse import urlparse


def get_extension(url: str) -> str:
    extension = Path(urlparse(url).path).suffix

    return extension.lower()


def main() -> None:
    parser = ArgumentParser()

    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)

    args = parser.parse_args()

    allowed_extensions = [".jpg", ".jpeg", ".png"]

    with open(args.input_path, "r") as f, open(args.output_path, "w") as o:
        for line in f:
            synset_id, urls_raw = line.strip().split("\t")
            urls = urls_raw.split(" ")

            urls = [url for url in urls if get_extension(url) in allowed_extensions]

            if urls:
                output = f"{synset_id}\t{urls[0]}\n"
                o.write(output)


if __name__ == "__main__":
    main()
