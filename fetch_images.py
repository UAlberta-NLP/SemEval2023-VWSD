'''
  Download a hierarchy of images specified by a custom JSON file

  python fetch_images.py
'''

import argparse
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import requests
import json
import os
from os.path import splitext, split, exists
import shutil
from utils import clean_ext
from PIL import Image
import io
from cairosvg import svg2png
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='./data/images.json')
parser.add_argument('--output', '-o', default='./data/bn_images')
args = parser.parse_args()

shared_dict = json.load(open(args.input))
shutil.rmtree(args.output, ignore_errors=True)
os.makedirs(args.output)

logging.basicConfig(level=logging.INFO)

headers = {'User-Agent': 'curl/7.68.0'}
def download_single_file(url, ext, dest_path):
  r = requests.get(url, stream=True, allow_redirects=True, headers=headers)
  if r.status_code == requests.codes.ok:
    content = r.content
    if ext in ['.jpg', '.png']:
      pi = Image.open(io.BytesIO(content))
    elif ext == '.svg':
      content = svg2png(bytestring=content, unsafe=True)
      pi = Image.open(io.BytesIO(content))
      ext = '.png'
      logging.info('Converted {} to {}'.format(url, dest_path))
    elif ext == '.gif':
      i = Image.open(io.BytesIO(content))
      f = io.BytesIO()
      i.save(f, format='png')
      content = f.getvalue()
      pi = Image.open(io.BytesIO(content))
      ext = '.png'
      logging.info('Converted {} to {}'.format(url, dest_path))
    else: 
      logging.info('Skipping {} because extension was {}'.format(url, splitext(dest_path)[-1]))
      return

    with open(dest_path, 'wb') as f:
      f.write(content)
      logging.info('Saving {} to {}'.format(url, dest_path))

def download_group(word, shared_dict=shared_dict):
  logging.info(f'Downloading images for {word}...')
  senses = shared_dict[word]
  target_dir = os.path.join(args.output, word)
  os.makedirs(target_dir)
  for sense in senses:
    urls = sense['images']
    id = sense['id']
    sense_dir = os.path.join(target_dir, id)
    os.makedirs(sense_dir)
    for idx, url in enumerate(urls):
      _, ext = splitext(url)
      ext = clean_ext(ext)
      file_path = os.path.join(sense_dir, f'{idx}{ext}')
      try:
        download_single_file(url, ext, file_path)
      except Exception as e:
        logging.error(f'Unable to save {url} to {file_path}: {e}')

words = list(shared_dict.keys())
results = ThreadPool(cpu_count()).imap_unordered(download_group, words)
for res in results:
  continue
