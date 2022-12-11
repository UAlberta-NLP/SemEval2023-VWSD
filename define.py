from ast import arg
from nltk.corpus import wordnet as wn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='data/text-wsd/mike.txt')
args = parser.parse_args()

for line in open(args.input).readlines():
  sense_key, *_ = line.split(' ')
  sense_key = sense_key.strip()
  print(wn.lemma_from_key(sense_key).synset().definition())