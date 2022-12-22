from argparse import ArgumentParser
from pathlib import Path
from urllib.parse import urlparse

import requests
from concurrent.futures import ThreadPoolExecutor


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)

    args = parser.parse_args()

    def download(data):
        synset_id, url = data
        headers = {
            "User-Agent": "<Talgat Omarov>/1.0 (omarov@ualberta.ca) bot for academic research project",
        }
        extension = Path(urlparse(url).path).suffix

        try:
            r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            if r.status_code == 200:
                with open(
                    args.output_folder / (synset_id + extension.lower()), "wb"
                ) as o:
                    o.write(r.content)
        except Exception:
            pass

    data = []
    with open(args.input_path, "r") as f:
        for line in f:
            synset_id, url = line.strip().split("\t")
            data.append((synset_id, url))

    with ThreadPoolExecutor(max_workers=64) as executor:
        executor.map(download, data)


if __name__ == "__main__":
    main()
