from argparse import ArgumentParser
from pathlib import Path
import requests


def is_url_image(image_url):
    headers = {
        "User-Agent": "<Talgat Omarov>/1.0 (omarov@ualberta.ca) bot for academic research project",
    }

    try: 
        r = requests.head(image_url, headers=headers, allow_redirects=True, timeout=10)
    except Exception:
        return False
    if 'content-type' in r.headers and r.headers["content-type"].startswith("image"):
        return True

    print(r.url)
    print(r.status_code)
    print(r.content)

    return False


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)

    args = parser.parse_args()

    with open(args.input_path, "r") as f, open(args.output_path, "w") as o:
        for line in f:
            lemma, synset_id, gold_url, candidate_urls_raw = line.strip().split("\t")
            candidate_urls = candidate_urls_raw.split(" ")

            if not is_url_image(gold_url):
                continue

            candidate_urls = [url for url in candidate_urls if is_url_image(url)]

            if len(candidate_urls) < 1:
                continue

            output_candidate_urls = " ".join(candidate_urls)

            output = [lemma, synset_id, gold_url, output_candidate_urls]

            ouput_text = "\t".join(output)

            o.write(f"{ouput_text}\n")


if __name__ == "__main__":
    main()
