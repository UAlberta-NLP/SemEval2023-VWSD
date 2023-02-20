from argparse import ArgumentParser
from glob import glob
import os

parser = ArgumentParser()
parser.add_argument('--inputs', nargs='+', default=['/local/storage/ogezi/babelpic/babelpic-gold/images'])
args = parser.parse_args()

map = {}
for path in args.inputs:
  file_names = os.listdir(os.path.join(path))
  for name in file_names:
    id = 'bn:' + name[3:].split('_')[0]
    if id not in map:
      map[id] = set()
    map[id].add(name)

cnt = 0
for k in map.keys():
  cnt += len(map[k])
  
mean = cnt / len(map)

print('Synsets:', len(map))
print('Images:', cnt)
print('Mean:', mean)