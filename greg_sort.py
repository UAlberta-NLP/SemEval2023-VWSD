import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-i', default='greg_search.log')
parser.add_argument('-o', default='greg_search.json')
args = parser.parse_args()

f = open(args.i).readlines()
perf_tup = [(l.split(': ')[0].strip(), float(l.split(': ')[1])) for l in f]
perf_tup = sorted(perf_tup, key=lambda x: x[1], reverse=True)

json.dump(dict(perf_tup), open(args.o, 'w+'), sort_keys=False, indent=2, ensure_ascii=True)