'''
  This shows how to enumerate the sense keys of a word
'''

from pathlib import Path
from nltk.corpus import wordnet as wn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--words', nargs='+', default=['anteater	marsupial anteater'], help='the word followed by a tab, and then, the context')
parser.add_argument('--input', '-i', type=Path, default=None)
args = parser.parse_args()

target_input = args.words if args.input is None else open(args.input).readlines()
words = [f.split('\t')[:2] for f in target_input]

for word, context in words:
  sense_key_to_gloss = dict([(max(x.lemmas(), key=lambda l: l.key().split('%')[0] == word).key(), x.definition()) for x in wn.synsets(word)])

  print(f'Word: "{word}"; Context: "{context}"')
  for i, (sk, gloss) in enumerate(sense_key_to_gloss.items()):
    print(f'\t{i + 1}. {sk} -> {gloss}')
  print()